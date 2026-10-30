// What the B-spline operation tests share: deterministic dyadic data, spline builders, and the
// exact-rational comparison of two splines (bspline_exact_eval.hpp), whose only rounding is the
// stored control points'.
#pragma once

#include "bspline_exact_eval.hpp"

#include <doctest/doctest.h>

#include <pantr/bspline/bspline.hpp>
#include <pantr/bspline/knot_insertion.hpp>
#include <pantr/bspline/knot_removal.hpp>
#include <pantr/tolerance.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace bspline_test {

namespace bs = pantr::bspline;
namespace oracle = bspline_oracle;
using oracle::Rational;

/// The knots, degree and periodicity of one direction.
struct Dir {
  std::vector<double> knots;
  int degree;
  bool periodic = false;
};

template <class T>
std::vector<T> cast_all(const std::vector<double>& v) {
  std::vector<T> out;
  out.reserve(v.size());
  for (const double x : v) {
    out.push_back(static_cast<T>(x));
  }
  return out;
}

template <class T>
std::span<const T> sp(const std::vector<T>& v) {
  return std::span<const T>(v);
}

/// Deterministic values in [-1, 1], multiples of 1/64 (exact in float).
inline std::vector<double> values(std::size_t count, unsigned seed) {
  std::vector<double> v(count);
  std::uint32_t s = 2463534242U + seed;
  for (double& x : v) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    x = static_cast<double>(static_cast<int>(s % 129U) - 64) / 64.0;
  }
  return v;
}

/// A knot vector with `degree + 1` copies of 0 and of 1 and the given interior knots.
inline std::vector<double> clamped(int degree, const std::vector<double>& interior) {
  std::vector<double> k(static_cast<std::size_t>(degree) + 1, 0.0);
  k.insert(k.end(), interior.begin(), interior.end());
  k.insert(k.end(), static_cast<std::size_t>(degree) + 1, 1.0);
  return k;
}

/// A uniform periodic knot vector of `n` intervals on [0, 1] with the ghost knots, at maximum
/// regularity (dyadic when `n` is a power of two).
inline std::vector<double> periodic_knots(int n, int degree) {
  std::vector<double> k;
  for (int i = -degree; i <= n + degree; ++i) {
    k.push_back(static_cast<double>(i) / static_cast<double>(n));
  }
  return k;
}

template <class T>
bs::BsplineSpace<T> space_of(const std::vector<Dir>& dirs) {
  std::vector<bs::BsplineSpace1D<T>> spaces;
  for (const Dir& d : dirs) {
    const std::vector<T> k = cast_all<T>(d.knots);
    spaces.emplace_back(sp(k), d.degree, d.periodic);
  }
  return bs::BsplineSpace<T>(std::move(spaces));
}

/// A B-spline over the given directions: `rank` coordinates, and for a rational one a weight in
/// [1, 2] (a multiple of 1/128) as the last.
template <class T>
bs::Bspline<T> make_spline(const std::vector<Dir>& dirs, std::size_t rank, bool rational,
                           unsigned seed) {
  bs::BsplineSpace<T> space = space_of<T>(dirs);
  std::size_t total = 1;
  for (const auto& s : space.spaces()) {
    total *= static_cast<std::size_t>(s.num_basis());
  }
  const std::size_t cols = rank + (rational ? 1 : 0);
  std::vector<double> flat = values(total * cols, seed);
  if (rational) {
    for (std::size_t i = 0; i < total; ++i) {
      flat[i * cols + rank] = 1.5 + flat[i * cols + rank] / 2.0;
    }
  }
  const std::vector<T> cp = cast_all<T>(flat);
  return bs::Bspline<T>(std::move(space), sp(cp), rational);
}

/// Insert the given knot values in one direction.
template <class T>
bs::Bspline<T> insert_in(const bs::Bspline<T>& b, std::size_t direction,
                         const std::vector<double>& add) {
  const std::vector<T> vals = cast_all<T>(add);
  std::vector<std::span<const T>> per(b.dim());
  per[direction] = sp(vals);
  return bs::insert_knots<T>(b, per);
}

/// Remove the given knot values in one direction.
template <class T>
bs::Bspline<T> remove_in(const bs::Bspline<T>& b, std::size_t direction,
                         const std::vector<double>& drop, std::optional<long long> num,
                         std::optional<double> tol) {
  const std::vector<T> vals = cast_all<T>(drop);
  std::vector<std::span<const T>> per(b.dim());
  per[direction] = sp(vals);
  return bs::remove_knots<T>(b, per, num, tol);
}

template <class T>
std::vector<T> knots_of(const bs::Bspline<T>& b, std::size_t direction) {
  const auto k = b.space().spaces()[direction].knots();
  return std::vector<T>(k.begin(), k.end());
}

/// The largest |a - b| over two equally long arrays.
template <class T>
double max_difference(std::span<const T> a, std::span<const T> b) {
  REQUIRE(a.size() == b.size());
  double worst = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    worst = (std::max)(worst, std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
  }
  return worst;
}

/// The conservative tier of `pantr::tolerance` times the control-point scale (at least 1): a long
/// accumulation, or a condition number up to about 1e3.
template <class T>
double tier_tolerance(const bs::Bspline<T>& b) {
  return static_cast<double>(pantr::tolerance::get_conservative<T>()) *
         (std::max)(1.0, oracle::max_abs_control_point<T>(b));
}

/// The largest exact difference of any stored coordinate between two splines, over the tensor
/// grid of the points `lo + (hi - lo) (2k + 1) / 17`, `k = 0..3`, of the first spline's domain
/// (none of them a knot of a dyadic mesh).
template <class T>
double exact_difference(const bs::Bspline<T>& a, const bs::Bspline<T>& b) {
  const std::size_t dim = a.dim();
  REQUIRE(b.dim() == dim);
  std::vector<std::vector<Rational>> axes(dim);
  for (std::size_t d = 0; d < dim; ++d) {
    const auto dom = a.space().spaces()[d].domain();
    const Rational lo = oracle::exact(static_cast<double>(dom.first));
    const Rational hi = oracle::exact(static_cast<double>(dom.second));
    for (int k = 0; k < 4; ++k) {
      axes[d].push_back(lo + (hi - lo) * Rational(2 * k + 1, 17));
    }
  }
  double worst = 0.0;
  std::vector<std::size_t> idx(dim, 0);
  for (;;) {
    std::vector<Rational> x;
    for (std::size_t d = 0; d < dim; ++d) {
      x.push_back(axes[d][idx[d]]);
    }
    const auto va = oracle::exact_value<T>(a, x);
    const auto vb = oracle::exact_value<T>(b, x);
    for (std::size_t c = 0; c < va.size(); ++c) {
      Rational diff = va[c] - vb[c];
      if (diff < 0) {
        diff = -diff;
      }
      worst = (std::max)(worst, static_cast<double>(diff));
    }
    std::size_t d = dim;
    bool done = true;
    while (d-- > 0) {
      if (++idx[d] < axes[d].size()) {
        done = false;
        break;
      }
      idx[d] = 0;
    }
    if (done) {
      return worst;
    }
  }
}

/// Whether two 1D spaces share one representation (an operation left the direction alone).
template <class Space>
bool same_space(const Space& a, const Space& b) {
  return a.rep() == b.rep();
}

/// The message of the exception a call throws, or a marker when it throws none.
inline std::string message_of(const std::function<void()>& call) {
  try {
    call();
  } catch (const std::exception& e) {
    return e.what();
  }
  return "<no exception>";
}

}  // namespace bspline_test
