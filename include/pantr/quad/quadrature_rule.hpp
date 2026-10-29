/// @file quadrature_rule.hpp
/// @brief `QuadratureRule`: points and weights of a rule on the unit cube `[0, 1]^ndim`.
#pragma once

#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/format.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/quad/rules_1d.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace pantr::quad {

/// A borrowed array of doubles with a runtime shape: how an argument whose rank is itself to be
/// validated reaches the validating constructor.
///
/// Attributes:
///     data (std::span<const double>): The elements, row-major.
///     shape (std::span<const std::size_t>): The extents; their product is `data.size()`.
struct DoubleArrayRef {
  std::span<const double> data;
  std::span<const std::size_t> shape;
};

/// A quadrature rule on the unit cube: points in `[0, 1]^ndim` and one weight per point.
///
/// An immutable value whose arrays are shared by every copy. The factories' rules have weights
/// summing to 1 up to the rounding of that sum.
///
/// Attributes:
///     rep_ (std::shared_ptr<const Rep>): The points `(num_points, ndim)` and weights.
class QuadratureRule {
 public:
  /// The shared representation.
  ///
  /// Attributes:
  ///     points (NdArray<double>): Extents `(num_points, ndim)`.
  ///     weights (NdArray<double>): Extents `(num_points,)`.
  struct Rep {
    NdArray<double> points;
    NdArray<double> weights;
  };

  /// Build and validate a rule, copying the arrays.
  ///
  /// Args:
  ///     points (DoubleArrayRef): Extents `(num_points, ndim)`, every coordinate in `[0, 1]`.
  ///     weights (DoubleArrayRef): Extents `(num_points,)`.
  ///
  /// Raises:
  ///     value_error: `points` is not 2-D, `weights` is not 1-D, their lengths differ, either is
  ///         empty, a value is not finite, or a point lies outside the unit cube.
  QuadratureRule(DoubleArrayRef points, DoubleArrayRef weights) {
    if (points.shape.size() != 2) {
      throw value_error("points must be 2D (num_points, ndim); got shape " + py_repr(points.shape) +
                        ".");
    }
    if (weights.shape.size() != 1) {
      throw value_error("weights must be 1D (num_points,); got shape " + py_repr(weights.shape) +
                        ".");
    }
    if (points.shape[0] == 0 || points.shape[1] == 0) {
      throw value_error("points must be non-empty; got shape " + py_repr(points.shape) + ".");
    }
    if (weights.shape[0] != points.shape[0]) {
      throw value_error("weights length " + std::to_string(weights.shape[0]) +
                        " must match the number of points " + std::to_string(points.shape[0]) +
                        ".");
    }
    for (const double v : points.data) {
      if (!std::isfinite(v)) {
        throw value_error("points must contain only finite values.");
      }
    }
    for (const double v : weights.data) {
      if (!std::isfinite(v)) {
        throw value_error("weights must contain only finite values.");
      }
    }
    for (const double v : points.data) {
      if (v < 0.0 || v > 1.0) {
        throw value_error("points must lie in the unit cube [0, 1]^ndim.");
      }
    }
    Rep rep{NdArray<double>({points.shape[0], points.shape[1]}),
            NdArray<double>({weights.shape[0]})};
    std::copy(points.data.begin(), points.data.end(), rep.points.data());
    std::copy(weights.data.begin(), weights.data.end(), rep.weights.data());
    rep_ = std::make_shared<const Rep>(std::move(rep));
  }

  /// Get the spatial dimension.
  ///
  /// Returns:
  ///     std::size_t: Number of axes, at least 1.
  [[nodiscard]] std::size_t ndim() const noexcept { return rep_->points.shape()[1]; }

  /// Get the number of points.
  ///
  /// Returns:
  ///     std::size_t: At least 1.
  [[nodiscard]] std::size_t num_points() const noexcept { return rep_->points.shape()[0]; }

  /// Get the points (a numeric view into the shared representation).
  ///
  /// Returns:
  ///     const NdArray<double>&: Extents `(num_points, ndim)`.
  [[nodiscard]] const NdArray<double>& points() const noexcept { return rep_->points; }

  /// Get the weights (a numeric view into the shared representation).
  ///
  /// Returns:
  ///     const NdArray<double>&: Extents `(num_points,)`.
  [[nodiscard]] const NdArray<double>& weights() const noexcept { return rep_->weights; }

  /// Get the shared representation, for an owner that must keep the arrays alive.
  ///
  /// Returns:
  ///     const std::shared_ptr<const Rep>&: The representation.
  [[nodiscard]] const std::shared_ptr<const Rep>& rep() const noexcept { return rep_; }

 private:
  std::shared_ptr<const Rep> rep_;
};

/// Build the tensor product of one-dimensional rules on `[0, 1]`.
///
/// Points in row-major order (the last axis varies fastest); each weight is the product of the
/// per-axis weights, multiplied in axis order.
///
/// Args:
///     rules (std::span<const std::pair<DoubleArrayRef, DoubleArrayRef>>): One `(nodes, weights)`
///         pair per axis, each 1-D of equal, non-zero length.
///
/// Returns:
///     QuadratureRule: The rule, `ndim == rules.size()`.
///
/// Raises:
///     value_error: No axis, an axis whose arrays are not 1-D or not matching non-empty arrays,
///         or (from the rule) a node outside `[0, 1]` or a non-finite value.
[[nodiscard]] inline QuadratureRule tensor_product_quadrature(
    std::span<const std::pair<DoubleArrayRef, DoubleArrayRef>> rules) {
  if (rules.empty()) {
    throw value_error("tensor_product_quadrature: rules must have at least one axis.");
  }
  std::size_t n = 1;
  for (std::size_t d = 0; d < rules.size(); ++d) {
    const auto& [nodes, weights] = rules[d];
    if (nodes.shape.size() != 1 || weights.shape.size() != 1) {
      throw value_error("tensor_product_quadrature: axis " + std::to_string(d) +
                        " nodes and weights must be 1D; got shapes " + py_repr(nodes.shape) +
                        " and " + py_repr(weights.shape) + ".");
    }
    if (nodes.shape[0] == 0 || nodes.shape[0] != weights.shape[0]) {
      throw value_error("tensor_product_quadrature: axis " + std::to_string(d) +
                        " needs matching non-empty (nodes, weights); got shapes " +
                        py_repr(nodes.shape) + " and " + py_repr(weights.shape) + ".");
    }
    n *= nodes.shape[0];
  }
  const std::size_t ndim = rules.size();
  std::vector<double> pts(n * ndim);
  std::vector<double> wts(n);
  std::vector<std::size_t> idx(ndim, 0);
  for (std::size_t p = 0; p < n; ++p) {
    double w = rules[0].second.data[idx[0]];
    for (std::size_t d = 0; d < ndim; ++d) {
      pts[p * ndim + d] = rules[d].first.data[idx[d]];
      if (d > 0) {
        w *= rules[d].second.data[idx[d]];
      }
    }
    wts[p] = w;
    for (std::size_t s = ndim; s-- > 0;) {
      if (++idx[s] < rules[s].first.shape[0]) {
        break;
      }
      idx[s] = 0;
    }
  }
  const std::size_t pshape[2] = {n, ndim};
  const std::size_t wshape[1] = {n};
  return QuadratureRule(DoubleArrayRef{pts, pshape}, DoubleArrayRef{wts, wshape});
}

/// Build the tensor-product Gauss-Legendre rule on the unit cube.
///
/// Args:
///     ndim (long long): Number of axes, at least 1.
///     npts (std::span<const long long>): Points per axis, `ndim` entries, each at least 1.
///
/// Returns:
///     QuadratureRule: Exact for tensor-product polynomials of per-axis degree `2 npts - 1`.
///
/// Raises:
///     value_error: `ndim < 1`, `npts.size() != ndim`, or a count below 1.
[[nodiscard]] inline QuadratureRule gauss_legendre_quadrature(long long ndim,
                                                              std::span<const long long> npts) {
  if (ndim < 1) {
    throw value_error("gauss_legendre_quadrature: ndim must be >= 1; got " + std::to_string(ndim) +
                      ".");
  }
  if (npts.size() != static_cast<std::size_t>(ndim)) {
    throw value_error("gauss_legendre_quadrature: npts must be a scalar or a length-" +
                      std::to_string(ndim) + " sequence; got length " +
                      std::to_string(npts.size()) + ".");
  }
  std::string listed = "(";
  bool bad = false;
  for (std::size_t i = 0; i < npts.size(); ++i) {
    bad = bad || npts[i] < 1;
    listed += (i > 0 ? ", " : "") + std::to_string(npts[i]);
  }
  listed += npts.size() == 1 ? ",)" : ")";
  if (bad) {
    throw value_error("gauss_legendre_quadrature: every npts entry must be >= 1; got " + listed +
                      ".");
  }
  std::vector<Rule1D<double>> rules1d;
  rules1d.reserve(npts.size());
  for (const long long n : npts) {
    rules1d.push_back(gauss_legendre_1d<double>(n));
  }
  std::vector<std::pair<DoubleArrayRef, DoubleArrayRef>> refs;
  refs.reserve(rules1d.size());
  for (const auto& r : rules1d) {
    refs.emplace_back(DoubleArrayRef{r.nodes.flat(), r.nodes.shape()},
                      DoubleArrayRef{r.weights.flat(), r.weights.shape()});
  }
  return tensor_product_quadrature(refs);
}

}  // namespace pantr::quad
