/// @file detect.hpp
/// @brief Detection of conforming interfaces among a set of B-spline patches.
///
/// `detect_interfaces` finds every pair of patch faces that coincide geometrically *and* carry
/// compatible tangential knot vectors, together with the axis correspondence relating them.
///
/// Only conforming interfaces are recognized. Two faces that occupy the same region of space but
/// disagree in degree or interior knots are rejected rather than approximated: accepting a nested
/// refinement on one side needs a projection the caller has to opt into, and silently pairing them
/// would produce control-point matches that are simply wrong.
#pragma once

#include "pantr/bspline/bspline.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/scalar.hpp"
#include "pantr/multipatch/interface.hpp"
#include "pantr/multipatch/match.hpp"
#include "pantr/parallel/parallel_for.hpp"
#include "pantr/parallel/work_model.hpp"
#include "pantr/tolerance.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace pantr::multipatch {

namespace detail {

/// The three tolerances an interface check needs, kept dimensionally separate.
///
/// Comparing all three quantities against one number is the classic silent error here: a patch
/// measured in millimetres would accept weight differences a metre apart, and a knot vector
/// normalized to `[0, 1]` has no length scale at all.
///
/// Attributes:
///     geometric (double): For control-point coordinates. The relative tolerance scaled by
///         `max(joint bounding-box diagonal, largest |coordinate|)`, so it tracks both the size of
///         the problem and where it sits.
///     parametric (double): For knot vectors already normalized to `[0, 1]`. Dimensionless, so it
///         is the bare relative tolerance.
///     weight (double): For rational weights. Scaled by the largest weight in play rather than by
///         any length.
struct Tolerances {
  double geometric;
  double parametric;
  double weight;
};

/// One candidate correspondence between the tangential axes of two faces.
///
/// Attributes:
///     axis_map (std::vector<long long>): Patch-b axis per tangential axis of face a.
///     flips (std::vector<bool>): Whether each of those axis pairs runs opposed.
struct Orientation {
  std::vector<long long> axis_map;
  std::vector<bool> flips;
};

/// Convert a tolerance to the storage type its comparison runs in.
///
/// A control-point difference of a `float` patch is formed and compared in `float` (as NumPy
/// compares a `float32` array with a Python float), so the tolerance is rounded to nearest `float`
/// first. A finite `double` beyond the `float` range converts as an infinity: the C++ conversion
/// is undefined there, and every difference is within an infinite tolerance.
///
/// Args:
///     tol (double): The tolerance.
///
/// Returns:
///     T: `tol` rounded to `T`; infinite beyond its range, NaN for NaN.
template <StorageFloat T>
[[nodiscard]] T narrow_tolerance(double tol) noexcept {
  if constexpr (std::is_same_v<T, double>) {
    return tol;
  } else {
    constexpr double largest = static_cast<double>((std::numeric_limits<T>::max)());
    if (std::isnan(tol)) {
      return std::numeric_limits<T>::quiet_NaN();
    }
    if (tol > largest) {
      return std::numeric_limits<T>::infinity();
    }
    if (tol < -largest) {
      return -std::numeric_limits<T>::infinity();
    }
    return static_cast<T>(tol);
  }
}

/// The larger of two values, a NaN in either winning (as NumPy's `max` reduction does).
template <class T>
[[nodiscard]] T max_nan(T x, T y) noexcept {
  if (std::isnan(x)) {
    return x;
  }
  if (std::isnan(y)) {
    return y;
  }
  return y > x ? y : x;
}

/// The smaller of two values, a NaN in either winning (as NumPy's `min` reduction does).
template <class T>
[[nodiscard]] T min_nan(T x, T y) noexcept {
  if (std::isnan(x)) {
    return x;
  }
  if (std::isnan(y)) {
    return y;
  }
  return y < x ? y : x;
}

/// What the detection reads of one patch, gathered once.
///
/// Attributes:
///     dim (std::size_t): Parametric dimension.
///     rank (std::size_t): Output rank, the weight excluded.
///     cp_size (std::size_t): Coordinates per control point, the weight included.
///     rational (bool): Whether the last coordinate is a weight.
///     degree (std::vector<int>): Degree per direction.
///     num_basis (std::vector<std::size_t>): Basis functions per direction.
///     stride (std::vector<std::size_t>): C-order strides of the control-point grid, in points.
///     knots (std::vector<std::vector<double>>): Each direction's knots mapped affinely onto
///         `[0, 1]` (all zero for a direction whose knots span nothing).
///     cps (std::span<const T>): The control points, row-major; views the patch's shared buffer,
///         which lives as long as the patch.
///     col_min (std::vector<T>): Per coordinate column, the smallest value (NaN if any is).
///     col_max (std::vector<T>): The largest value (NaN if any is).
///     col_abs_max (std::vector<T>): The largest magnitude (NaN if any value is).
template <StorageFloat T>
struct PatchData {
  std::size_t dim;
  std::size_t rank;
  std::size_t cp_size;
  bool rational;
  std::vector<int> degree;
  std::vector<std::size_t> num_basis;
  std::vector<std::size_t> stride;
  std::vector<std::vector<double>> knots;
  std::span<const T> cps;
  std::vector<T> col_min;
  std::vector<T> col_max;
  std::vector<T> col_abs_max;
};

/// Gather what the detection reads of a patch.
///
/// Args:
///     b (const bspline::Bspline<T>&): The patch.
///
/// Returns:
///     PatchData<T>: The gathered data; its `cps` views `b`'s buffer.
template <StorageFloat T>
[[nodiscard]] PatchData<T> make_patch_data(const bspline::Bspline<T>& b) {
  PatchData<T> d;
  d.dim = b.dim();
  d.rank = b.rank();
  d.cp_size = b.cp_size();
  d.rational = b.rational();
  const bspline::BsplineSpace<T> space = b.space();
  for (const auto& s : space.spaces()) {
    d.degree.push_back(s.degree());
    d.num_basis.push_back(static_cast<std::size_t>(s.num_basis()));
    // The knots as doubles, then `(k - k[0]) / (k[-1] - k[0])` over the whole vector, ghost knots
    // included.
    const std::span<const T> raw = s.knots();
    const double first = static_cast<double>(raw.front());
    const double span = static_cast<double>(raw.back()) - first;
    std::vector<double> norm(raw.size(), 0.0);
    if (span > 0.0) {
      for (std::size_t i = 0; i < raw.size(); ++i) {
        norm[i] = (static_cast<double>(raw[i]) - first) / span;
      }
    }
    d.knots.push_back(std::move(norm));
  }
  d.stride = c_strides(d.num_basis);
  d.cps = b.control_points();

  const std::size_t n_points = d.cps.size() / d.cp_size;
  d.col_min.assign(d.cps.begin(), d.cps.begin() + static_cast<std::ptrdiff_t>(d.cp_size));
  d.col_max = d.col_min;
  d.col_abs_max.resize(d.cp_size);
  for (std::size_t c = 0; c < d.cp_size; ++c) {
    d.col_abs_max[c] = std::abs(d.col_min[c]);
  }
  for (std::size_t p = 1; p < n_points; ++p) {
    for (std::size_t c = 0; c < d.cp_size; ++c) {
      const T v = d.cps[p * d.cp_size + c];
      d.col_min[c] = min_nan(d.col_min[c], v);
      d.col_max[c] = max_nan(d.col_max[c], v);
      d.col_abs_max[c] = max_nan(d.col_abs_max[c], static_cast<T>(std::abs(v)));
    }
  }
  return d;
}

/// Refuse a pair whose second patch has fewer control-point coordinates than the first's rank.
///
/// The joint tolerance reads the first `rank_a` coordinates of both patches; a second patch with
/// fewer has nothing to read. (Patches of different ranks are otherwise tolerated, and never
/// match, as the first corner comparison finds.)
///
/// Args:
///     a (const PatchData<T>&): The first patch's data.
///     index_a (long long): Its index.
///     b (const PatchData<T>&): The second patch's data.
///     index_b (long long): Its index.
///
/// Raises:
///     value_error: `b.cp_size < a.rank`.
template <StorageFloat T>
void check_comparable(const PatchData<T>& a, long long index_a, const PatchData<T>& b,
                      long long index_b) {
  if (b.cp_size < a.rank) {
    throw value_error("Patch " + std::to_string(index_b) + " has " + std::to_string(b.cp_size) +
                      " control-point coordinates per point, fewer than the rank " +
                      std::to_string(a.rank) + " of patch " + std::to_string(index_a) +
                      "; their coordinates cannot be compared.");
  }
}

/// Derive the geometric, parametric and weight tolerances for a patch pair.
///
/// `relative` is not `space.tolerance`: that is an *absolute* parametric length carrying the
/// patch's own knot magnitude, and none of the three quantities compared here is a parametric
/// coordinate of a single patch. What is wanted is the bare dimensionless factor, which each leg
/// below scales by the magnitude that applies to it.
///
/// The diagonal alone is not the scale a coordinate comparison is relative to. Two small patches
/// meeting at x = 1e6 have a diagonal of order one while the difference of two of their
/// coordinates is formed from numbers of order 1e6, so it carries an absolute error of eps * 1e6
/// however short the diagonal is; grading against the diagonal there asks for agreement eight
/// orders below the noise floor. The largest coordinate magnitude is therefore taken alongside it
/// and the larger wins -- the same rule, for the same reason, as `knot_scale` in the B-spline
/// layer and as the physical half of `geometric_scale` in point inversion
/// (`include/pantr/bspline/locate.hpp`). That one carries a third length for what its
/// parametrization resolves, which has no counterpart here: this grades a distance between two
/// physical corner coordinates.
///
/// A degenerate joint bounding box (both patches a single point, at the origin) leaves no scale to
/// borrow, so the geometric tolerance falls back to the bare relative tolerance rather than to
/// zero.
///
/// Both patches' first `a.rank` coordinate columns are read, as the Python reference did. The
/// extent is formed in the storage type `T` (as NumPy does for a `float32` array) and measured in
/// `double`.
///
/// Args:
///     a (const PatchData<T>&): First patch.
///     b (const PatchData<T>&): Second patch.
///     relative (double): The dimensionless relative tolerance for the pair.
///
/// Returns:
///     Tolerances: The three derived absolute tolerances.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed): `b.cp_size >= a.rank`
///     (`check_comparable`).
template <StorageFloat T>
[[nodiscard]] Tolerances joint_tolerances(const PatchData<T>& a, const PatchData<T>& b,
                                          double relative) noexcept {
  double diagonal = 0.0;
  double magnitude = 0.0;
  for (std::size_t c = 0; c < a.rank; ++c) {
    const T hi = max_nan(a.col_max[c], b.col_max[c]);
    const T lo = min_nan(a.col_min[c], b.col_min[c]);
    const double extent = static_cast<double>(static_cast<T>(hi - lo));
    // `hypot` rather than the square root of a sum of squares: no intermediate overflow.
    diagonal = c == 0 ? std::abs(extent) : std::hypot(diagonal, extent);
    const auto abs_max = static_cast<double>(max_nan(a.col_abs_max[c], b.col_abs_max[c]));
    magnitude = c == 0 ? abs_max : max_nan(magnitude, abs_max);
  }
  const double scale = magnitude > diagonal ? magnitude : diagonal;
  const double geometric = scale > 0.0 ? relative * scale : relative;

  double weight_scale = 1.0;
  if (a.rational || b.rational) {
    double largest = 0.0;
    bool have = false;
    for (const PatchData<T>* d : {&a, &b}) {
      if (!d->rational) {
        continue;
      }
      const auto w = static_cast<double>(d->col_abs_max[d->cp_size - 1]);
      largest = (!have || w > largest) ? w : largest;
      have = true;
    }
    weight_scale = largest > 1.0 ? largest : 1.0;
  }
  return Tolerances{geometric, relative, relative * weight_scale};
}

/// Get the geometric coordinates of a face's corner control points.
///
/// The weight is not part of them: a rational patch's coordinates are the homogeneous ones, not
/// projected.
///
/// Args:
///     d (const PatchData<T>&): The patch.
///     face (long long): Face id.
///
/// Returns:
///     std::vector<double>: `2^(dim - 1)` corners of `d.rank` coordinates each, row-major, in an
///     unspecified order (callers compare corner *sets*).
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat T>
[[nodiscard]] std::vector<double> face_corner_coords(const PatchData<T>& d, long long face) {
  const auto [own, side] = face_axis_side(face);
  const auto axis = static_cast<std::size_t>(own);
  const std::size_t slab = side == 0 ? 0 : d.num_basis[axis] - 1;
  const std::vector<long long> tang = tangential_axes(face, d.dim);
  const std::size_t n_tang = tang.size();
  const std::size_t n_corners = std::size_t{1} << n_tang;
  std::vector<double> corners(n_corners * d.rank);
  for (std::size_t m = 0; m < n_corners; ++m) {
    std::size_t flat = slab * d.stride[axis];
    for (std::size_t k = 0; k < n_tang; ++k) {
      const auto axis_t = static_cast<std::size_t>(tang[k]);
      const bool high = ((m >> (n_tang - 1 - k)) & 1U) != 0;
      flat += (high ? d.num_basis[axis_t] - 1 : 0) * d.stride[axis_t];
    }
    for (std::size_t c = 0; c < d.rank; ++c) {
      corners[m * d.rank + c] = static_cast<double>(d.cps[flat * d.cp_size + c]);
    }
  }
  return corners;
}

/// Check whether every corner of one face has a partner within `tol` among the other face's
/// corners, and conversely.
///
/// A cheap pre-filter before the full verification: it rejects the vast majority of face pairs
/// without touching the interior control points, and it deliberately ignores ordering, since the
/// ordering is what the orientation search determines. Every corner of each face must have a
/// partner on the other; a bijection is not required here because coincident corners are
/// legitimate on degenerate patches.
///
/// Args:
///     corners_a (std::span<const double>): First face's corners, row-major.
///     rank_a (std::size_t): Coordinates per corner of the first face.
///     corners_b (std::span<const double>): Second face's corners.
///     rank_b (std::size_t): Coordinates per corner of the second face.
///     tol (double): Absolute geometric tolerance.
///
/// Returns:
///     bool: True if the two corner sets coincide within `tol`; false for faces whose corners
///     have different numbers of coordinates (different ranks), which cannot coincide. A corner
///     with a NaN coordinate has no partner.
[[nodiscard]] inline bool corner_sets_match(std::span<const double> corners_a, std::size_t rank_a,
                                            std::span<const double> corners_b, std::size_t rank_b,
                                            double tol) noexcept {
  if (rank_a != rank_b) {
    return false;
  }
  const std::size_t n = corners_a.size() / rank_a;
  const auto distance = [&](std::size_t i, std::size_t j) {
    double sum = 0.0;
    for (std::size_t c = 0; c < rank_a; ++c) {
      const double diff = corners_a[i * rank_a + c] - corners_b[j * rank_a + c];
      sum += diff * diff;
    }
    return std::sqrt(sum);
  };
  for (std::size_t i = 0; i < n; ++i) {
    bool found = false;
    for (std::size_t j = 0; j < n && !found; ++j) {
      found = distance(i, j) <= tol;
    }
    if (!found) {
      return false;
    }
  }
  for (std::size_t j = 0; j < n; ++j) {
    bool found = false;
    for (std::size_t i = 0; i < n && !found; ++i) {
      found = distance(i, j) <= tol;
    }
    if (!found) {
      return false;
    }
  }
  return true;
}

/// Check that the tangential knot vectors agree under the given correspondence.
///
/// Args:
///     a (const PatchData<T>&): First patch.
///     face_a (long long): Face id on it.
///     b (const PatchData<T>&): Second patch.
///     orientation (const Orientation&): Candidate axis correspondence.
///     tol (double): Dimensionless tolerance for knots normalized to `[0, 1]`.
///
/// Returns:
///     bool: True if every matched direction agrees in degree and in normalized knot vector, the
///     second one reversed (`1 - k` read backwards) where the pair is flipped.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat T>
[[nodiscard]] bool knots_agree(const PatchData<T>& a, long long face_a, const PatchData<T>& b,
                               const Orientation& orientation, double tol) {
  const std::vector<long long> tang_a = tangential_axes(face_a, a.dim);
  for (std::size_t k = 0; k < tang_a.size(); ++k) {
    const auto axis_a = static_cast<std::size_t>(tang_a[k]);
    const auto axis_b = static_cast<std::size_t>(orientation.axis_map[k]);
    if (a.degree[axis_a] != b.degree[axis_b]) {
      return false;
    }
    const std::vector<double>& knots_a = a.knots[axis_a];
    const std::vector<double>& knots_b = b.knots[axis_b];
    if (knots_a.size() != knots_b.size()) {
      return false;
    }
    const std::size_t n = knots_a.size();
    const bool flip = orientation.flips[k];
    for (std::size_t i = 0; i < n; ++i) {
      const double other = flip ? 1.0 - knots_b[n - 1 - i] : knots_b[i];
      if (!(std::abs(knots_a[i] - other) <= tol)) {
        return false;
      }
    }
  }
  return true;
}

/// Check that every matched control point pair coincides, weights included.
///
/// The pairing comes from `build_matched_cps`, the arithmetic `match_face_cps` runs, so this cannot
/// drift from the index arithmetic a caller will later use to unify dofs. A pair of faces whose
/// matched directions hold different numbers of functions has no pairing: no match.
///
/// The first `a.rank` coordinates of both patches are compared in the storage type `T`, against
/// the geometric tolerance rounded to `T`. The weight check runs when the *first* patch is
/// rational, and compares the last coordinate of each patch (for a polynomial second patch that is
/// a coordinate), as the Python reference did.
///
/// Args:
///     a (const PatchData<T>&): First patch.
///     face_a (long long): Face id on it.
///     b (const PatchData<T>&): Second patch.
///     face_b (long long): Face id on it.
///     orientation (const Orientation&): Candidate axis correspondence.
///     tolerances (const Tolerances&): Geometric and weight tolerances.
///
/// Returns:
///     bool: True if all matched coordinates agree within the geometric tolerance and all matched
///     weights within the weight tolerance.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed): `b.cp_size >= a.rank`.
template <StorageFloat T>
[[nodiscard]] bool cps_agree(const PatchData<T>& a, long long face_a, const PatchData<T>& b,
                             long long face_b, const Orientation& orientation,
                             const Tolerances& tolerances) {
  if (!tangential_counts_agree(a.num_basis, face_a, b.num_basis, orientation.axis_map)) {
    return false;
  }
  const MatchedCps m = build_matched_cps(a.num_basis, face_a, b.num_basis, face_b,
                                         orientation.axis_map, orientation.flips);
  const T geometric = narrow_tolerance<T>(tolerances.geometric);
  for (std::size_t i = 0; i < m.a.size(); ++i) {
    const T* pa = a.cps.data() + static_cast<std::size_t>(m.a[i]) * a.cp_size;
    const T* pb = b.cps.data() + static_cast<std::size_t>(m.b[i]) * b.cp_size;
    for (std::size_t c = 0; c < a.rank; ++c) {
      if (!(std::abs(static_cast<T>(pa[c] - pb[c])) <= geometric)) {
        return false;
      }
    }
  }
  if (a.rational) {
    const T weight = narrow_tolerance<T>(tolerances.weight);
    for (std::size_t i = 0; i < m.a.size(); ++i) {
      const T wa = a.cps[static_cast<std::size_t>(m.a[i]) * a.cp_size + a.cp_size - 1];
      const T wb = b.cps[static_cast<std::size_t>(m.b[i]) * b.cp_size + b.cp_size - 1];
      if (!(std::abs(static_cast<T>(wa - wb)) <= weight)) {
        return false;
      }
    }
  }
  return true;
}

/// Enumerate every axis correspondence a face pair could have.
///
/// There are `(dim - 1)! * 2^(dim - 1)` of them: 1 in 1D, 2 in 2D, 8 in 3D. Enumerating and
/// verifying is what makes detection robust on symmetric faces, where the corner correspondence
/// alone does not determine the orientation. The order is that of the permutations of face b's
/// tangential axes in lexicographic order, each followed by the flip patterns counting from none
/// to all with the first axis the most significant, so the identity permutation without flips
/// comes first and the first passing candidate wins.
///
/// Args:
///     face_b (long long): Face id on the second patch.
///     dim (std::size_t): Parametric dimension.
///
/// Returns:
///     std::vector<Orientation>: Every candidate correspondence.
[[nodiscard]] inline std::vector<Orientation> orientation_candidates(long long face_b,
                                                                     std::size_t dim) {
  std::vector<long long> permutation = tangential_axes(face_b, dim);  // increasing: the first one
  const std::size_t n = permutation.size();
  std::vector<Orientation> out;
  do {
    for (std::size_t m = 0; m < (std::size_t{1} << n); ++m) {
      std::vector<bool> flips(n);
      for (std::size_t k = 0; k < n; ++k) {
        flips[k] = ((m >> (n - 1 - k)) & 1U) != 0;
      }
      out.push_back(Orientation{permutation, std::move(flips)});
    }
  } while (std::next_permutation(permutation.begin(), permutation.end()));
  return out;
}

/// Refuse a set of patches that do not share one parametric dimension.
///
/// Args:
///     patches (std::span<const bspline::Bspline<T>>): The patches, at least one.
///
/// Returns:
///     std::size_t: The shared dimension.
///
/// Raises:
///     value_error: The patches differ in parametric dimension.
template <StorageFloat T>
[[nodiscard]] std::size_t common_dim(std::span<const bspline::Bspline<T>> patches) {
  std::vector<std::size_t> dims;
  for (const auto& p : patches) {
    dims.push_back(p.dim());
  }
  std::sort(dims.begin(), dims.end());
  dims.erase(std::unique(dims.begin(), dims.end()), dims.end());
  if (dims.size() != 1) {
    std::string list = "[";
    for (std::size_t i = 0; i < dims.size(); ++i) {
      list += (i > 0 ? ", " : "") + std::to_string(dims[i]);
    }
    list += ']';
    throw value_error("All patches must share one parametric dimension; got " + list);
  }
  return dims.front();
}

/// Find the conforming interfaces among a set of patches, with the relative tolerance of each
/// patch given.
///
/// The relative tolerance of a pair is `tol` when given, else the larger of its two patches'
/// entries of `relative_per_patch`. The homogeneous `detect_interfaces` passes the strict preset of
/// `T` for every patch; the binding passes each patch's own dtype's when it has converted patches
/// of different dtypes to one.
///
/// Args:
///     patches (std::span<const bspline::Bspline<T>>): The patches.
///     relative_per_patch (std::span<const double>): The default relative tolerance of each
///         patch, `patches.size()` entries.
///     tol (std::optional<double>): Dimensionless relative tolerance override.
///
/// Returns:
///     std::vector<Interface>: The interfaces found, sorted by
///     `(patch_a, face_a, patch_b, face_b)`.
///
/// Raises:
///     value_error: As `detect_interfaces`.
///
/// Note:
///     `relative_per_patch` is assumed to have one entry per patch (no validation performed).
template <StorageFloat T>
[[nodiscard]] std::vector<Interface> detect_interfaces(std::span<const bspline::Bspline<T>> patches,
                                                       std::span<const double> relative_per_patch,
                                                       std::optional<double> tol) {
  if (patches.empty()) {
    return {};
  }
  const std::size_t dim = common_dim(patches);
  const std::size_t n_patches = patches.size();

  std::vector<PatchData<T>> data;
  data.reserve(n_patches);
  for (const auto& p : patches) {
    data.push_back(make_patch_data(p));
  }

  // The first pair, in the order the loop below visits them, whose second patch cannot be compared
  // with the first: found here so the error does not depend on which thread meets one first.
  // `smallest[q]` is the fewest coordinates per point among patches q, q + 1, ...
  std::vector<std::size_t> smallest(n_patches + 1, (std::numeric_limits<std::size_t>::max)());
  for (std::size_t q = n_patches; q-- > 0;) {
    smallest[q] = (std::min)(smallest[q + 1], data[q].cp_size);
  }
  for (std::size_t p = 0; p + 1 < n_patches; ++p) {
    if (smallest[p + 1] < data[p].rank) {
      std::size_t q = p + 1;
      while (data[q].cp_size >= data[p].rank) {
        ++q;
      }
      check_comparable(data[p], static_cast<long long>(p), data[q], static_cast<long long>(q));
    }
  }

  const std::size_t faces_per_patch = 2 * dim;
  const std::size_t n_sides = n_patches * faces_per_patch;
  std::vector<std::vector<double>> corners(n_sides);
  for (std::size_t s = 0; s < n_sides; ++s) {
    corners[s] =
        face_corner_coords(data[s / faces_per_patch], static_cast<long long>(s % faces_per_patch));
  }
  std::vector<std::vector<Orientation>> candidates(faces_per_patch);
  for (std::size_t f = 0; f < faces_per_patch; ++f) {
    candidates[f] = orientation_candidates(static_cast<long long>(f), dim);
  }

  // One item per first side; its partners are the later sides. The sides are numbered
  // `patch * 2 * dim + face`, so ascending item and then ascending partner is ascending
  // `(patch_a, face_a, patch_b, face_b)`, and the per-item lists concatenate in that order whatever
  // the number of threads.
  std::vector<std::vector<Interface>> per_side(n_sides);
  // Work estimate: an admitted heuristic. A pair that fails (nearly all do) costs the corner
  // distances, `corners^2` of them over `rank` coordinates; the partners of an item number
  // `n_sides - 1 - i`, `(n_sides - 1) / 2` on average over the loop. The orientation search of
  // the few pairs whose corners coincide is not counted.
  std::size_t max_rank = 1;
  for (const PatchData<T>& d : data) {
    max_rank = (std::max)(max_rank, d.rank);
  }
  const double n_corners = std::ldexp(1.0, static_cast<int>(dim) - 1);
  const double pairs_per_item = static_cast<double>(n_sides - 1) / 2.0;
  const double ops_per_item =
      (std::max)(1.0, pairs_per_item * n_corners * n_corners * static_cast<double>(max_rank));
  parallel::parallel_for(
      n_sides, parallel::WorkEstimate{ops_per_item}, [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
          const std::size_t patch_a = i / faces_per_patch;
          const auto face_a = static_cast<long long>(i % faces_per_patch);
          const PatchData<T>& a = data[patch_a];
          for (std::size_t j = i + 1; j < n_sides; ++j) {
            const std::size_t patch_b = j / faces_per_patch;
            const auto face_b = static_cast<long long>(j % faces_per_patch);
            const PatchData<T>& b = data[patch_b];
            const double relative =
                tol ? *tol : (std::max)(relative_per_patch[patch_a], relative_per_patch[patch_b]);
            const Tolerances tolerances = joint_tolerances(a, b, relative);
            if (!corner_sets_match(corners[i], a.rank, corners[j], b.rank, tolerances.geometric)) {
              continue;
            }
            for (const Orientation& orientation : candidates[static_cast<std::size_t>(face_b)]) {
              if (!knots_agree(a, face_a, b, orientation, tolerances.parametric)) {
                continue;
              }
              if (!cps_agree(a, face_a, b, face_b, orientation, tolerances)) {
                continue;
              }
              per_side[i].emplace_back(static_cast<long long>(patch_a), face_a,
                                       static_cast<long long>(patch_b), face_b,
                                       orientation.axis_map, orientation.flips);
              break;
            }
          }
        }
      });

  std::vector<Interface> found;
  for (auto& list : per_side) {
    for (auto& item : list) {
      found.push_back(std::move(item));
    }
  }
  return found;
}

}  // namespace detail

/// Find the conforming interfaces among a set of patches.
///
/// Every pair of faces in canonical order is considered, including two faces of the same patch,
/// which is how a closed ring is found. A pair becomes an interface only when both conditions
/// hold: all matched control points coincide, and the matched tangential knot vectors agree after
/// affine normalization to `[0, 1]`. A face pair that occupies the same space but disagrees in
/// degree or interior knots is *not* an interface here.
///
/// The pairs are tested on the process pool (`parallel_for`); the result does not depend on the
/// number of threads. Patches of different control-point widths are tolerated (they never match)
/// unless a later patch has fewer coordinates per point than an earlier one has output
/// coordinates, which raises.
///
/// Args:
///     patches (std::span<const bspline::Bspline<T>>): The patches to inspect, possibly none. All
///         must share the same parametric dimension.
///     tol (std::optional<double>): Dimensionless relative tolerance override. Defaults to the
///         strict preset of `T`; it is scaled by the larger of the joint bounding-box diagonal and
///         the largest coordinate magnitude for coordinates, and used bare for knots.
///
/// Returns:
///     std::vector<Interface>: The interfaces found, sorted by
///     `(patch_a, face_a, patch_b, face_b)`; empty for no patches.
///
/// Raises:
///     value_error: The patches do not all share the same parametric dimension; or a patch has
///         fewer control-point coordinates per point than the rank of an earlier one.
template <StorageFloat T>
[[nodiscard]] std::vector<Interface> detect_interfaces(std::span<const bspline::Bspline<T>> patches,
                                                       std::optional<double> tol = std::nullopt) {
  const std::vector<double> relative(patches.size(),
                                     static_cast<double>(tolerance::get_strict<T>()));
  return detail::detect_interfaces<T>(patches, relative, tol);
}

/// Check that an interface actually holds for a set of patches.
///
/// Applies the same two conditions detection requires: every matched control point pair coincides,
/// and the tangential knot vectors agree after normalization.
///
/// Args:
///     patches (std::span<const bspline::Bspline<T>>): The patches the interface refers to.
///     iface (const Interface&): The interface to verify.
///     tol (std::optional<double>): Dimensionless relative tolerance override. Defaults to the
///         strict preset of `T`.
///
/// Returns:
///     bool: True if the interface holds.
///
/// Raises:
///     index_error: The interface refers to a patch index that does not exist.
///     value_error: A patch's parametric dimension differs from the interface's; or the second
///         patch has fewer control-point coordinates per point than the first one's rank.
template <StorageFloat T>
[[nodiscard]] bool verify_interface(std::span<const bspline::Bspline<T>> patches,
                                    const Interface& iface,
                                    std::optional<double> tol = std::nullopt) {
  for (const long long index : {iface.patch_a(), iface.patch_b()}) {
    if (!(0 <= index && static_cast<unsigned long long>(index) < patches.size())) {
      throw index_error("Interface patch index " + std::to_string(index) + " is out of range for " +
                        std::to_string(patches.size()) + " patches");
    }
  }
  const auto& patch_a = patches[static_cast<std::size_t>(iface.patch_a())];
  const auto& patch_b = patches[static_cast<std::size_t>(iface.patch_b())];
  for (const auto* p : {&patch_a, &patch_b}) {
    if (p->dim() != iface.dim()) {
      throw value_error("Interface implies dim=" + std::to_string(iface.dim()) +
                        " but the patches are " + std::to_string(p->dim()) + "D");
    }
  }
  const detail::PatchData<T> a = detail::make_patch_data(patch_a);
  const detail::PatchData<T> b = detail::make_patch_data(patch_b);
  detail::check_comparable(a, iface.patch_a(), b, iface.patch_b());
  const detail::Orientation orientation{iface.axis_map(), iface.flips()};
  const double relative = tol ? *tol : static_cast<double>(tolerance::get_strict<T>());
  const detail::Tolerances tolerances = detail::joint_tolerances(a, b, relative);
  if (!detail::knots_agree(a, iface.face_a(), b, orientation, tolerances.parametric)) {
    return false;
  }
  return detail::cps_agree(a, iface.face_a(), b, iface.face_b(), orientation, tolerances);
}

}  // namespace pantr::multipatch
