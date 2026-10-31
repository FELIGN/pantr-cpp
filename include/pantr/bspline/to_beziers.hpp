/// @file to_beziers.hpp
/// @brief Conversion of a `Bspline` to Bézier form: the single Bézier of a one-element B-spline
///        (`to_bezier`) and the Bézier patches of every element (`to_beziers`).
///
/// The decomposition applies the Bézier extraction operators of each direction to the control
/// points, one direction after the other (Borden et al., "Isogeometric finite element data
/// structures based on Bézier extraction of NURBS", 2011): along direction `d` the control points
/// of element `e` are `C_e^T P_local`, `P_local` the `degree + 1` control points that carry the
/// element's functions, each product being a `mode_product`.
#pragma once

#include "pantr/bezier/bezier.hpp"
#include "pantr/bezier/detail/along_axis.hpp"
#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/detail/mode_product.hpp"
#include "pantr/bspline/extraction.hpp"
#include "pantr/bspline/knot_insertion.hpp"
#include "pantr/bspline/space.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"
#include "pantr/parallel/parallel_for.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace pantr::bspline {

/// The Bézier patches of a B-spline, one per element, in C order over the elements.
///
/// Attributes:
///     extents (std::vector<std::size_t>): The elements per direction.
///     beziers (std::vector<bezier::Bezier<T>>): `prod(extents)` patches; the one of element
///         `(e_0, ..., e_{dim-1})` is at the C-order index of that multi-index.
template <StorageFloat T>
struct BezierDecomposition {
  std::vector<std::size_t> extents;
  std::vector<bezier::Bezier<T>> beziers;
};

namespace detail {

/// Refuse a B-spline that is more than one polynomial piece.
inline constexpr const char* more_than_one_element =
    "B-spline has more than one element per direction and cannot be represented as a single "
    "Bézier.";

/// Apply the Bézier extraction operators of one direction to a control net, element by element.
///
/// Args:
///     space (const BsplineSpace1D<T>&): The direction's space (open ends or not; periodic
///         directions are converted before).
///     x (std::span<const T>): The net, row-major over `x_extents`.
///     x_extents (std::span<const std::size_t>): Its extents; along `mode` the space's number of
///         basis functions.
///     mode (std::size_t): The direction.
///
/// Returns:
///     NdArray<T>: The net with `mode`'s extent replaced by `intervals * (degree + 1)`: the Bézier
///     control points of the elements stacked along the direction.
template <StorageFloat T>
[[nodiscard]] NdArray<T> extract_direction(const BsplineSpace1D<T>& space, std::span<const T> x,
                                           std::span<const std::size_t> x_extents,
                                           std::size_t mode) {
  const auto order = static_cast<std::size_t>(space.degree()) + 1;
  const auto n_el = static_cast<std::size_t>(space.num_intervals());
  const NdArray<T> ops = bezier_extraction_operators<T>(space);
  const std::span<const std::int64_t> first = space.first_basis_per_interval();
  std::vector<std::size_t> y_extents(x_extents.begin(), x_extents.end());
  y_extents[mode] = n_el * order;
  NdArray<T> y(y_extents, for_overwrite);
  const bezier::detail::AxisSplit around = bezier::detail::split_around(x_extents, mode);
  const std::size_t x_extent = x_extents[mode];
  const std::size_t y_extent = y_extents[mode];
  // One element is `order * order * outer * inner` multiply-adds; the elements write disjoint
  // parts of `y`.
  const double ops_per_element = 2.0 * static_cast<double>(order) * static_cast<double>(order) *
                                 static_cast<double>(around.outer) *
                                 static_cast<double>(around.inner);
  const T* ops_data = ops.data();
  const T* x_data = x.data();
  T* y_data = y.data();
  parallel::parallel_for(n_el, parallel::WorkEstimate{(std::max)(ops_per_element, 1.0)},
                         [&](std::size_t begin, std::size_t end) {
                           for (std::size_t e = begin; e < end; ++e) {
                             // `C_e^T`: the stored operator, transposed.
                             mode_product_kernel<T>(
                                 ops_data + e * order * order, order, order, order, true,
                                 x_data + static_cast<std::size_t>(first[e]) * around.inner,
                                 x_extent, y_data + e * order * around.inner, y_extent,
                                 around.outer, around.inner);
                           }
                         });
  return y;
}

}  // namespace detail

/// Convert a B-spline with a single polynomial piece to a Bézier.
///
/// A B-spline with Bézier-like knots (open ends, `degree + 1` functions in each direction) has the
/// Bézier's control points as its own. Any other single-element B-spline (periodic or with
/// unclamped knots) is converted to open form first, which builds fresh control points that the
/// Bézier then shares.
///
/// Args:
///     b (const Bspline<T>&): The B-spline.
///     copy (bool): Copy the control points into the Bézier. With `false` a B-spline with
///         Bézier-like knots shares its buffer with the Bézier (the Bézier keeps it alive); the
///         buffer is immutable either way. Defaults to true.
///
/// Returns:
///     bezier::Bezier<T>: The Bézier, rational as `b` is.
///
/// Raises:
///     value_error: `b` has more than one element in some direction.
template <StorageFloat T>
[[nodiscard]] bezier::Bezier<T> to_bezier(const Bspline<T>& b, bool copy = true) {
  const BsplineSpace<T> whole = b.space();
  const std::vector<std::size_t> shape(b.shape().begin(), b.shape().end());
  if (whole.has_bezier_like_knots()) {
    if (copy) {
      return bezier::Bezier<T>(b.control_points(), shape, b.rational());
    }
    // An aliasing pointer: the Bézier points into the B-spline's buffer and keeps its whole
    // representation alive.
    return bezier::Bezier<T>(std::shared_ptr<const T[]>(b.rep(), b.rep()->data.get()), shape,
                             b.rational());
  }
  const bool all_open =
      std::all_of(whole.spaces().begin(), whole.spaces().end(),
                  [](const auto& s) { return s.has_open_knots() && !s.periodic(); });
  if (all_open) {
    throw value_error(detail::more_than_one_element);
  }
  const Bspline<T> open = to_open_bspline<T>(b);
  if (!open.space().has_bezier_like_knots()) {
    throw value_error(detail::more_than_one_element);
  }
  // `open` owns fresh control points that nothing else sees: the Bézier shares them.
  return bezier::Bezier<T>(std::shared_ptr<const T[]>(open.rep(), open.rep()->data.get()), shape,
                           b.rational());
}

/// Decompose a B-spline into the Bézier patches of its elements.
///
/// Periodic directions are converted to open form first. Along each direction the Bézier
/// extraction operators are applied to the control points (all elements at once, the elements in
/// parallel), then the patches are cut out of the result.
///
/// Args:
///     b (const Bspline<T>&): The B-spline.
///
/// Returns:
///     BezierDecomposition<T>: The patches in C order over the elements, each with its own control
///     points, rational as `b` is.
///
/// Raises:
///     capacity_error: The decomposition is too large to index.
template <StorageFloat T>
[[nodiscard]] BezierDecomposition<T> to_beziers(const Bspline<T>& input) {
  const BsplineSpace<T> input_space = input.space();
  const bool any_periodic = std::any_of(input_space.spaces().begin(), input_space.spaces().end(),
                                        [](const auto& s) { return s.periodic(); });
  const Bspline<T> b = any_periodic ? to_open_bspline<T>(input) : input;
  const BsplineSpace<T> whole = b.space();
  const std::size_t dim = b.dim();
  const std::size_t cp = b.cp_size();

  std::vector<std::size_t> shape(b.shape().begin(), b.shape().end());
  std::span<const T> current = b.control_points();
  NdArray<T> owned = detail::empty_net<T>();
  for (std::size_t d = 0; d < dim; ++d) {
    owned = detail::extract_direction<T>(whole.spaces()[d], current, shape, d);
    shape[d] = owned.shape()[d];
    current = owned.flat();
  }

  std::vector<std::size_t> extents(dim);
  std::vector<std::size_t> orders(dim);
  for (std::size_t d = 0; d < dim; ++d) {
    extents[d] = static_cast<std::size_t>(whole.spaces()[d].num_intervals());
    orders[d] = static_cast<std::size_t>(whole.spaces()[d].degree()) + 1;
  }
  const std::size_t n_elements = checked_product(extents);
  const std::size_t points_per_element = checked_product(orders);
  // Row strides of the stacked net, in units of rows of `cp` values.
  std::vector<std::size_t> stride(dim, 1);
  for (std::size_t d = dim - 1; d-- > 0;) {
    stride[d] = stride[d + 1] * shape[d + 1];
  }
  std::vector<std::size_t> patch_shape(orders);
  patch_shape.push_back(cp);

  BezierDecomposition<T> out{extents, {}};
  out.beziers.reserve(n_elements);
  std::vector<std::size_t> element(dim, 0);
  const std::size_t run = orders[dim - 1] * cp;  // the last direction's points are contiguous
  const std::size_t runs = points_per_element / orders[dim - 1];
  std::vector<std::size_t> local(dim, 0);
  for (std::size_t n = 0; n < n_elements; ++n) {
    NdArray<T> net(patch_shape, for_overwrite);
    std::fill(local.begin(), local.end(), 0);
    for (std::size_t r = 0; r < runs; ++r) {
      std::size_t row = 0;
      for (std::size_t d = 0; d < dim; ++d) {
        row += (element[d] * orders[d] + local[d]) * stride[d];
      }
      std::copy_n(current.data() + row * cp, run, net.data() + r * run);
      // Advance the local multi-index over every direction but the last.
      for (std::size_t d = dim - 1; d-- > 0;) {
        if (++local[d] < orders[d]) {
          break;
        }
        local[d] = 0;
      }
    }
    out.beziers.emplace_back(std::move(net), b.rational());
    for (std::size_t d = dim; d-- > 0;) {
      if (++element[d] < extents[d]) {
        break;
      }
      element[d] = 0;
    }
  }
  return out;
}

}  // namespace pantr::bspline
