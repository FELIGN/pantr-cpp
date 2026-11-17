// What the `pantr::cad` tests share: point evaluation into doubles, grids of parameters, and the
// machine epsilon of a storage type. No oracle lives here; each test file states its own.
#pragma once

#include <doctest/doctest.h>

#include <pantr/bspline/bspline.hpp>
#include <pantr/bspline/evaluate.hpp>
#include <pantr/core/mdspan.hpp>

#include <cstddef>
#include <limits>
#include <span>
#include <vector>

namespace cad_test {

namespace bs = pantr::bspline;

/// The machine epsilon of a storage type, as a double.
template <class T>
constexpr double eps() {
  return static_cast<double>(std::numeric_limits<T>::epsilon());
}

/// `n` equally spaced values from `lo` to `hi` inclusive, `lo + (hi - lo) i / (n - 1)`.
inline std::vector<double> linspace(double lo, double hi, std::size_t n) {
  std::vector<double> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = n == 1 ? lo : lo + (hi - lo) * static_cast<double>(i) / static_cast<double>(n - 1);
  }
  return out;
}

/// The tensor grid of per-direction samples as rows, the last direction varying fastest.
inline std::vector<double> grid(const std::vector<std::vector<double>>& axes) {
  std::size_t total = 1;
  for (const auto& a : axes) {
    total *= a.size();
  }
  std::vector<double> rows;
  rows.reserve(total * axes.size());
  std::vector<std::size_t> idx(axes.size(), 0);
  for (std::size_t n = 0; n < total; ++n) {
    for (std::size_t d = 0; d < axes.size(); ++d) {
      rows.push_back(axes[d][idx[d]]);
    }
    for (std::size_t d = axes.size(); d-- > 0;) {
      if (++idx[d] < axes[d].size()) {
        break;
      }
      idx[d] = 0;
    }
  }
  return rows;
}

/// Evaluate a B-spline at points given as rows of doubles (rounded to `T`), returning the
/// `n * rank` values as doubles. A rational B-spline is projected.
template <class T>
std::vector<double> evaluate_at(const bs::Bspline<T>& b, const std::vector<double>& rows) {
  std::vector<T> pts;
  pts.reserve(rows.size());
  for (const double r : rows) {
    pts.push_back(static_cast<T>(r));
  }
  const std::size_t n = rows.size() / b.dim();
  const pantr::NdArray<T> values =
      bs::evaluate<T, T>(b, pantr::mdview<const T, 2>(pts.data(), n, b.dim()));
  std::vector<double> out;
  out.reserve(values.size());
  for (const T v : values.flat()) {
    out.push_back(static_cast<double>(v));
  }
  return out;
}

/// The control points of a B-spline as doubles.
template <class T>
std::vector<double> control_points_of(const bs::Bspline<T>& b) {
  const std::span<const T> cp = b.control_points();
  return std::vector<double>(cp.begin(), cp.end());
}

/// The knots of one direction as doubles.
template <class T>
std::vector<double> knots_as_double(const bs::Bspline<T>& b, std::size_t direction) {
  const auto k = b.space().spaces()[direction].knots();
  return std::vector<double>(k.begin(), k.end());
}

}  // namespace cad_test
