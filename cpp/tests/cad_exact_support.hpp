// What the Coons and join tests share: curves and surfaces from flat control points, the exact
// rational value of a B-spline (bspline_exact_eval.hpp) as the oracle, and the matching of a
// refusal message, including the gap, tolerance and scale numbers it reports. The numbers a test
// expects are computed from the data and `numeric_limits`, and printed by `snprintf`, a route that
// shares nothing with `py_format_exp`.
#pragma once

#include "bspline_test_support.hpp"

#include <doctest/doctest.h>

#include <pantr/bspline/bspline.hpp>
#include <pantr/bspline/space.hpp>
#include <pantr/bspline/space_1d.hpp>
#include <pantr/core/errors.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace cad_test {

namespace bs = pantr::bspline;
namespace oracle = bspline_oracle;
using oracle::Rational;

using Pt = std::array<double, 3>;

/// A curve of the given degree over a clamped knot vector with `interior` knots, from flat
/// control points (`width` per point; the last a weight when `rational`).
template <class T>
bs::Bspline<T> curve_flat(int degree, const std::vector<double>& interior,
                          const std::vector<double>& flat, bool rational = false) {
  const std::vector<T> k = bspline_test::cast_all<T>(bspline_test::clamped(degree, interior));
  std::vector<bs::BsplineSpace1D<T>> spaces;
  spaces.emplace_back(std::span<const T>(k), degree);
  const std::vector<T> cp = bspline_test::cast_all<T>(flat);
  return bs::Bspline<T>(bs::BsplineSpace<T>(std::move(spaces)), std::span<const T>(cp), rational);
}

inline std::vector<double> flatten(const std::vector<Pt>& pts) {
  std::vector<double> flat;
  for (const Pt& p : pts) {
    flat.insert(flat.end(), p.begin(), p.end());
  }
  return flat;
}

/// A non-rational rank-3 curve through the given control points.
template <class T>
bs::Bspline<T> curve_of(int degree, const std::vector<double>& interior,
                        const std::vector<Pt>& pts) {
  return curve_flat<T>(degree, interior, flatten(pts));
}

template <class T>
bs::Bspline<T> line(const Pt& a, const Pt& b) {
  return curve_of<T>(1, {}, {a, b});
}

/// A rational line from homogeneous end points `(w x, w y, w z, w)`.
template <class T>
bs::Bspline<T> hom_line(const std::array<double, 4>& a, const std::array<double, 4>& b) {
  return curve_flat<T>(1, {}, {a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]}, true);
}

/// A surface of bidegree `(deg_u, deg_v)` over clamped knots, from flat control points row-major
/// over `(u, v, coefficient)`.
template <class T>
bs::Bspline<T> surface_flat(int deg_u, const std::vector<double>& in_u, int deg_v,
                            const std::vector<double>& in_v, const std::vector<double>& flat,
                            bool rational = false) {
  const std::vector<T> ku = bspline_test::cast_all<T>(bspline_test::clamped(deg_u, in_u));
  const std::vector<T> kv = bspline_test::cast_all<T>(bspline_test::clamped(deg_v, in_v));
  std::vector<bs::BsplineSpace1D<T>> spaces;
  spaces.emplace_back(std::span<const T>(ku), deg_u);
  spaces.emplace_back(std::span<const T>(kv), deg_v);
  const std::vector<T> cp = bspline_test::cast_all<T>(flat);
  return bs::Bspline<T>(bs::BsplineSpace<T>(std::move(spaces)), std::span<const T>(cp), rational);
}

/// The B-spline over the same space with the control points replaced.
template <class T>
bs::Bspline<T> with_control_points(const bs::Bspline<T>& b, const std::vector<double>& flat) {
  const std::vector<T> cp = bspline_test::cast_all<T>(flat);
  return bs::Bspline<T>(b.space(), std::span<const T>(cp), b.rational());
}

/// The exact value of `b` (every stored coordinate: homogeneous for a rational one) at `x`.
template <class T>
std::vector<Rational> value_at(const bs::Bspline<T>& b, const std::vector<double>& x) {
  std::vector<Rational> xs;
  for (const double v : x) {
    xs.push_back(oracle::exact(v));
  }
  return oracle::exact_value<T>(b, std::span<const Rational>(xs));
}

inline double max_difference(const std::vector<Rational>& a, const std::vector<Rational>& b) {
  REQUIRE(a.size() == b.size());
  double worst = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    Rational d = a[i] - b[i];
    if (d < 0) {
      d = -d;
    }
    worst = (std::max)(worst, static_cast<double>(d));
  }
  return worst;
}

/// The message of the `value_error` a call throws, or a marker when it throws none. Another
/// exception type propagates and fails the test.
inline std::string refusal(const std::function<void()>& call) {
  try {
    call();
  } catch (const pantr::value_error& e) {
    return e.what();
  }
  return "<no value_error>";
}

inline bool has(const std::string& s, const std::string& sub) {
  return s.find(sub) != std::string::npos;
}

/// `%.3e`, as Python's `{x:.3e}`.
inline std::string fmt3(double x) {
  std::array<char, 64> buf{};
  std::snprintf(buf.data(), buf.size(), "%.3e", x);
  return buf.data();
}

/// The conservative tier of a type: `4096 eps`.
template <class T>
double tier_of() {
  return 4096.0 * static_cast<double>(std::numeric_limits<T>::epsilon());
}

/// The numbers a mismatch message reports: `(gap X, above Y at <population> scale Z)`.
struct Reported {
  double gap;
  double tol;
  double scale;
  std::string population;
};

inline Reported reported_numbers(const std::string& message) {
  const std::size_t g = message.find("(gap ");
  REQUIRE_MESSAGE(g != std::string::npos, message);
  const std::string tail = message.substr(g + 5);
  char* end = nullptr;
  Reported r{};
  r.gap = std::strtod(tail.c_str(), &end);
  const std::string after_gap(end);
  REQUIRE_MESSAGE(after_gap.compare(0, 8, ", above ") == 0, message);
  r.tol = std::strtod(after_gap.c_str() + 8, &end);
  const std::string after_tol(end);
  REQUIRE_MESSAGE(after_tol.compare(0, 4, " at ") == 0, message);
  const std::string rest = after_tol.substr(4);
  const std::size_t s = rest.find(" scale ");
  REQUIRE_MESSAGE(s != std::string::npos, message);
  r.population = rest.substr(0, s);
  r.scale = std::strtod(rest.c_str() + s + 7, nullptr);
  return r;
}

}  // namespace cad_test
