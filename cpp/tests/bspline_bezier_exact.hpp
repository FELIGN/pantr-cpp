// Exact-rational Bernstein form for the Bézier extraction tests: the Bernstein basis at a rational
// point, and the value of a `Bezier` from its stored control points. The only rounding in a
// comparison against these is the stored control points' (and the operator's) own.
#pragma once

#include "bspline_exact_eval.hpp"

#include <pantr/bezier/bezier.hpp>

#include <cstddef>
#include <limits>
#include <vector>

namespace bspline_test {

using bspline_oracle::Rational;

/// The unit roundoff of the storage type.
template <class T>
double unit_roundoff() {
  return 0.5 * static_cast<double>(std::numeric_limits<T>::epsilon());
}

inline Rational binomial_exact(int n, int k) {
  Rational r = 1;
  for (int i = 1; i <= k; ++i) {
    r = r * (n - k + i) / i;
  }
  return r;
}

inline Rational bernstein_exact(int n, int k, const Rational& x) {
  Rational r = binomial_exact(n, k);
  for (int i = 0; i < k; ++i) {
    r *= x;
  }
  for (int i = 0; i < n - k; ++i) {
    r *= (1 - x);
  }
  return r;
}

/// The exact value (all coordinates) of a Bézier at `x`, from its stored control points.
template <class T>
std::vector<Rational> bezier_exact(const pantr::bezier::Bezier<T>& b,
                                   const std::vector<Rational>& x) {
  const std::size_t dim = b.dim();
  const std::size_t cols = b.cp_size();
  std::vector<std::vector<Rational>> basis(dim);
  for (std::size_t d = 0; d < dim; ++d) {
    const int p = static_cast<int>(b.degree(d));
    for (int k = 0; k <= p; ++k) {
      basis[d].push_back(bernstein_exact(p, k, x[d]));
    }
  }
  std::vector<Rational> out(cols, Rational(0));
  std::vector<std::size_t> idx(dim, 0);
  const auto cp = b.control_points();
  for (;;) {
    Rational w = 1;
    std::size_t row = 0;
    for (std::size_t d = 0; d < dim; ++d) {
      w *= basis[d][idx[d]];
      row = row * b.shape()[d] + idx[d];
    }
    for (std::size_t c = 0; c < cols; ++c) {
      out[c] += w * bspline_oracle::exact(static_cast<double>(cp[row * cols + c]));
    }
    std::size_t d = dim;
    bool done = true;
    while (d-- > 0) {
      if (++idx[d] < b.shape()[d]) {
        done = false;
        break;
      }
      idx[d] = 0;
    }
    if (done) {
      return out;
    }
  }
}

}  // namespace bspline_test
