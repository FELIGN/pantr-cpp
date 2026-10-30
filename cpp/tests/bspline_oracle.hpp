// An exact-rational B-spline oracle for the B-spline tests: the basis from its recursive definition
// (not the triangle of Piegl and Tiller A2.2 and A2.3 that the library runs), over the whole knot
// vector, in exact arithmetic. Every knot and every point is a float or a double, hence a dyadic
// rational, so the oracle's values carry no rounding at all and a test compares against them with
// a tolerance that belongs to the library alone.
//
//   N_{i,0}(x)  = 1 if i is the span of x, else 0
//   N_{i,q}(x)  = (x - u_i) / (u_{i+q} - u_i) N_{i,q-1}(x) + (u_{i+q+1} - x) / (u_{i+q+1} -
//   u_{i+1})
//                 N_{i+1,q-1}(x), a term whose denominator is zero being zero (0 / 0 = 0)
//   D^k N_{i,q} = the same recurrence differentiated k times (Leibniz: D^k[(x - a) f] =
//                 (x - a) D^k f + k D^{k-1} f), so no derivative formula of the library is reused.
#pragma once

#include <pantr/core/exact_rational.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

namespace bspline_oracle {

using pantr::exact::Integer;
using pantr::exact::Rational;

/// `2^k` for `k >= 0`, by multiplication (GCC 14 at -O3 reports a false -Wstringop-overflow inside
/// Boost's shift of an integer whose size it can see, the same as `exact_rational.hpp` notes).
inline Integer power_of_two(unsigned k) {
  Integer p = 1;
  while (k >= 30) {
    p *= Integer(1U << 30);
    k -= 30;
  }
  p *= Integer(1U << k);
  return p;
}

/// The exact value of a finite double (a float converts to a double first, exactly).
inline Rational exact(double x) {
  int e = 0;
  const double m = std::frexp(x, &e);  // x = m 2^e, |m| in [0.5, 1), or 0
  const auto mant = static_cast<std::int64_t>(std::ldexp(m, 53));  // an integer of at most 53 bits
  const int shift = e - 53;
  Integer num = mant;
  Integer den = 1;
  if (shift >= 0) {
    num *= power_of_two(static_cast<unsigned>(shift));
  } else {
    den = power_of_two(static_cast<unsigned>(-shift));
  }
  return Rational(num, den);
}

/// The exact knots of a knot vector.
template <class T>
std::vector<Rational> exact_knots(std::span<const T> knots) {
  std::vector<Rational> u;
  u.reserve(knots.size());
  for (const T k : knots) {
    u.push_back(exact(static_cast<double>(k)));
  }
  return u;
}

/// Whether `got` is finite and within `tol` of `want`, both compared exactly.
inline bool within(double got, const Rational& want, double tol) {
  if (!std::isfinite(got)) {
    return false;
  }
  Rational diff = exact(got) - want;
  if (diff < 0) {
    diff = -diff;
  }
  return diff <= exact(tol);
}

/// The span the library must evaluate `x` on: the first `j` in `[degree, size - degree - 2]` with
/// `x < u[j + 1]`, else the last (so the right end of the domain is evaluated on the last span, and
/// a point left of the domain on the first). A linear scan, not a binary search plus clamps.
inline std::size_t span_of(const std::vector<Rational>& u, int degree, const Rational& x) {
  const std::size_t last = u.size() - static_cast<std::size_t>(degree) - 2;
  for (std::size_t j = static_cast<std::size_t>(degree); j < last; ++j) {
    if (x < u[j + 1]) {
      return j;
    }
  }
  return last;
}

/// The basis functions, and their derivatives, of one knot vector at one point.
class BasisAt {
 public:
  BasisAt(std::vector<Rational> knots, int degree, const Rational& x)
      : u_(std::move(knots)), x_(x), degree_(degree), span_(span_of(u_, degree, x)) {}

  /// The span `x` is evaluated on.
  [[nodiscard]] std::size_t span() const { return span_; }

  /// The `k`-th derivative of `N_{i, degree}` at `x`, on the polynomial piece of `span()`.
  Rational deriv(std::size_t i, int k) { return d(i, degree_, k); }

 private:
  Rational d(std::size_t i, int q, int k) {
    const auto key = std::make_tuple(i, q, k);
    const auto it = memo_.find(key);
    if (it != memo_.end()) {
      return it->second;
    }
    Rational result = 0;
    if (q == 0) {
      result = (k == 0 && i == span_) ? 1 : 0;
    } else {
      const Rational den1 = u_[i + static_cast<std::size_t>(q)] - u_[i];
      if (den1 != 0) {
        Rational t = (x_ - u_[i]) * d(i, q - 1, k);
        if (k > 0) {
          t += k * d(i, q - 1, k - 1);
        }
        result += t / den1;
      }
      const Rational den2 = u_[i + static_cast<std::size_t>(q) + 1] - u_[i + 1];
      if (den2 != 0) {
        Rational t = (u_[i + static_cast<std::size_t>(q) + 1] - x_) * d(i + 1, q - 1, k);
        if (k > 0) {
          t -= k * d(i + 1, q - 1, k - 1);
        }
        result += t / den2;
      }
    }
    memo_.emplace(key, result);
    return result;
  }

  std::vector<Rational> u_;
  Rational x_;
  int degree_;
  std::size_t span_;
  std::map<std::tuple<std::size_t, int, int>, Rational> memo_;
};

}  // namespace bspline_oracle
