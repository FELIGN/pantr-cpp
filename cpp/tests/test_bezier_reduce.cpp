// Bezier degree reduction, degree minimisation and the reduction-operator process table.
//
// Oracles. Exact rational: the reduction operators are re-derived at degree <= 8 from the
// definition (the L^2-closest polynomial of the lower degree, with both end values kept for the
// interpolating operator), with the Gram and cross-Gram integrals taken term by term on the
// monomial expansion of the Bernstein polynomials and the normal equations solved by an elimination
// written here, so nothing is shared with the library's Farouki-Rajan Gram formula, its elevation
// matrix or its solver. Analytic: the cubic-to-quadratic row (-1/4, 3/4, 3/4, -1/4) derived by hand
// below, the mean (row of 1/(p + 1)) for a reduction to degree 0, and the L^2 errors of t^3, t^2
// and t, whose integrals are evaluated by hand. Round trip: reducing an elevated curve returns
// it (R E = I). Invariants: unit first and last rows, bit-exact end values, rows summing to 1.
//
// Rounding. The library rounds each exact operator entry once, to nearest. The oracle checks that
// directly: an entry passes if no neighbouring double is closer to the exact rational (distances
// taken exactly), so a truncation, a double rounding or a rounding of a pre-rounded value fails.
#include <doctest/doctest.h>

#include <pantr/bezier/bezier.hpp>
#include <pantr/bezier/elevate.hpp>
#include <pantr/bezier/reduce.hpp>
#include <pantr/core/errors.hpp>
#include <pantr/core/exact_rational.hpp>
#include <pantr/core/ndarray.hpp>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace pb = pantr::bezier;
using pantr::NdArray;
using pantr::exact::Integer;
using pantr::exact::Rational;

namespace {

using Matrix = std::vector<std::vector<Rational>>;

constexpr double u_double = std::numeric_limits<double>::epsilon() / 2.0;

template <class T>
constexpr double unit_roundoff() {
  return static_cast<double>(std::numeric_limits<T>::epsilon()) / 2.0;
}

Rational rat(long long num, long long den = 1) { return Rational(Integer(num), Integer(den)); }

double to_double(const Rational& q) { return q.convert_to<double>(); }

Rational abs_of(const Rational& q) { return q < 0 ? Rational(-q) : q; }

Integer binomial(int n, int k) {
  Integer r = 1;
  for (int i = 1; i <= k; ++i) {
    r = r * (n - k + i) / i;
  }
  return r;
}

// B_i^n(t) = C(n, i) t^i (1 - t)^(n - i) as monomial coefficients (index = power of t).
std::vector<Rational> bernstein_monomials(int n, int i) {
  std::vector<Rational> m(static_cast<std::size_t>(n) + 1, Rational(0));
  for (int k = i; k <= n; ++k) {
    const Rational sign = ((k - i) % 2 == 0) ? rat(1) : rat(-1);
    m[static_cast<std::size_t>(k)] = sign * Rational(binomial(n, i) * binomial(n - i, k - i));
  }
  return m;
}

// The integral over [0, 1] of B_i^m B_j^n, term by term on the monomial expansions.
Rational inner_product(int m, int i, int n, int j) {
  const std::vector<Rational> a = bernstein_monomials(m, i);
  const std::vector<Rational> b = bernstein_monomials(n, j);
  Rational sum = 0;
  for (std::size_t k = 0; k < a.size(); ++k) {
    for (std::size_t l = 0; l < b.size(); ++l) {
      sum += a[k] * b[l] / Rational(static_cast<long long>(k + l + 1));
    }
  }
  return sum;
}

Matrix gram_matrix(int n) {
  Matrix g(static_cast<std::size_t>(n) + 1, std::vector<Rational>(static_cast<std::size_t>(n) + 1));
  for (int i = 0; i <= n; ++i) {
    for (int j = 0; j <= n; ++j) {
      g[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = inner_product(n, i, n, j);
    }
  }
  return g;
}

// Solve a x = b for square a by Gauss-Jordan elimination with the first nonzero pivot.
Matrix solve(Matrix a, Matrix b) {
  const std::size_t n = a.size();
  for (std::size_t col = 0; col < n; ++col) {
    std::size_t piv = col;
    while (piv < n && a[piv][col] == 0) {
      ++piv;
    }
    REQUIRE(piv < n);  // a singular system would be an error of the oracle itself
    std::swap(a[col], a[piv]);
    std::swap(b[col], b[piv]);
    const Rational d = a[col][col];
    for (auto& v : a[col]) {
      v /= d;
    }
    for (auto& v : b[col]) {
      v /= d;
    }
    for (std::size_t r = 0; r < n; ++r) {
      if (r == col || a[r][col] == 0) {
        continue;
      }
      const Rational f = a[r][col];
      for (std::size_t k = 0; k < n; ++k) {
        a[r][k] -= f * a[col][k];
      }
      for (std::size_t k = 0; k < b[r].size(); ++k) {
        b[r][k] -= f * b[col][k];
      }
    }
  }
  return b;
}

// The exact reduction operator from degree p to p - t: rows q + 1, columns p + 1.
Matrix exact_operator(int p, int t, bool interpolating) {
  const int q = p - t;
  const auto qs = static_cast<std::size_t>(q);
  const auto ps = static_cast<std::size_t>(p);
  const Matrix gram = gram_matrix(q);
  Matrix cross(qs + 1, std::vector<Rational>(ps + 1));
  for (int i = 0; i <= q; ++i) {
    for (int j = 0; j <= p; ++j) {
      cross[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = inner_product(q, i, p, j);
    }
  }
  if (!interpolating || q == 0) {
    return solve(gram, cross);  // the unconstrained projection; q == 0 cannot honour two ends
  }
  Matrix rows(qs + 1, std::vector<Rational>(ps + 1, Rational(0)));
  rows[0][0] = 1;
  rows[qs][ps] = 1;
  if (q >= 2) {
    Matrix a(qs - 1, std::vector<Rational>(qs - 1));
    Matrix b(qs - 1, std::vector<Rational>(ps + 1));
    for (std::size_t i = 1; i < qs; ++i) {
      for (std::size_t k = 1; k < qs; ++k) {
        a[i - 1][k - 1] = gram[i][k];
      }
      b[i - 1] = cross[i];
      b[i - 1][0] -= gram[i][0];    // the kept end value q_0 = c_0
      b[i - 1][ps] -= gram[i][qs];  // the kept end value q_q = c_p
    }
    const Matrix solved = solve(a, b);
    for (std::size_t i = 1; i < qs; ++i) {
      rows[i] = solved[i - 1];
    }
  }
  return rows;
}

// The largest absolute row sum of the exact operator.
double lebesgue_of(const Matrix& m) {
  double worst = 0.0;
  for (const auto& row : m) {
    Rational s = 0;
    for (const Rational& v : row) {
      s += abs_of(v);
    }
    worst = (std::max)(worst, to_double(s));
  }
  return worst;
}

// Whether `g` is a correctly rounded (to nearest) double of `q`: no neighbour of `g` is strictly
// closer to `q`. Distances are exact, in rationals.
bool is_nearest_double(double g, const Rational& q) {
  const Rational dg = abs_of(Rational(g) - q);
  const double lo = std::nextafter(g, -std::numeric_limits<double>::infinity());
  const double hi = std::nextafter(g, std::numeric_limits<double>::infinity());
  return dg <= abs_of(Rational(lo) - q) && dg <= abs_of(Rational(hi) - q);
}

// Each entry is the exact entry rounded once to nearest. Bit identity is the right bar here: both
// sides are one correctly rounded value of the same rational, with no accumulated rounding.
void check_matches_exact(const NdArray<double>& op, const Matrix& exact) {
  REQUIRE(op.shape().size() == 2);
  REQUIRE(op.shape()[0] == exact.size());
  REQUIRE(op.shape()[1] == exact[0].size());
  const std::size_t cols = exact[0].size();
  for (std::size_t i = 0; i < exact.size(); ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      const double got = op.data()[i * cols + j];
      CAPTURE(i);
      CAPTURE(j);
      CAPTURE(got);
      CAPTURE(to_double(exact[i][j]));
      if (exact[i][j] == 0) {
        CHECK(got == 0.0);  // an exact zero stays exactly zero
      } else {
        CHECK(is_nearest_double(got, exact[i][j]));
      }
    }
  }
}

struct Lcg {
  std::uint64_t s = 20260921;
  double next() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>(s >> 11) / 4503599627370496.0 - 1.0;  // [-1, 1)
  }
};

template <class T>
std::vector<T> random_vec(Lcg& rng, std::size_t n) {
  std::vector<T> v(n);
  for (T& x : v) {
    x = static_cast<T>(rng.next());
  }
  return v;
}

template <class T>
pb::Bezier<T> make_bezier(const std::vector<T>& ctrl, std::vector<std::size_t> shape,
                          bool rational = false) {
  return pb::Bezier<T>(std::span<const T>(ctrl), std::move(shape), rational);
}

template <class T>
std::vector<T> copy_of(const pb::Bezier<T>& b) {
  return {b.control_points().begin(), b.control_points().end()};
}

template <class T>
double largest_magnitude(const std::vector<T>& v) {
  double big = 0.0;
  for (const T x : v) {
    big = (std::max)(big, std::fabs(static_cast<double>(x)));
  }
  return big;
}

template <class T>
bool same_bits(T a, T b) {
  if constexpr (sizeof(T) == 4) {
    return std::bit_cast<std::uint32_t>(a) == std::bit_cast<std::uint32_t>(b);
  } else {
    return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
  }
}

// The exact L^2([0, 1]^d) norm of a tensor Bernstein polynomial given by double coefficients
// (shape = orders per direction, the rank components stored last), as double.
double exact_l2_norm(std::span<const double> coeffs, const std::vector<std::size_t>& orders,
                     std::size_t rank) {
  std::vector<Matrix> grams;
  for (const std::size_t o : orders) {
    grams.push_back(gram_matrix(static_cast<int>(o) - 1));
  }
  std::size_t n = 1;
  for (const std::size_t o : orders) {
    n *= o;
  }
  const auto decode = [&](std::size_t flat) {
    std::vector<std::size_t> idx(orders.size());
    for (std::size_t d = orders.size(); d-- > 0;) {
      idx[d] = flat % orders[d];
      flat /= orders[d];
    }
    return idx;
  };
  Rational total = 0;
  for (std::size_t r = 0; r < rank; ++r) {
    for (std::size_t a = 0; a < n; ++a) {
      for (std::size_t b = 0; b < n; ++b) {
        const std::vector<std::size_t> ia = decode(a);
        const std::vector<std::size_t> ib = decode(b);
        Rational w = Rational(coeffs[a * rank + r]) * Rational(coeffs[b * rank + r]);
        for (std::size_t d = 0; d < orders.size(); ++d) {
          w *= grams[d][ia[d]][ib[d]];
        }
        total += w;
      }
    }
  }
  return std::sqrt(to_double(total));
}

}  // namespace

// ---- The operators against the exact oracle
// ------------------------------------------------------

TEST_CASE("reduce operators: the interpolating operator equals the exact-rational oracle") {
  // Bug caught: a wrong Gram entry, a wrong elevation matrix, an off-by-one in the interior block,
  // a rounding that is not to nearest, in any (degree, decrement) pair with degree <= 8.
  for (int p = 1; p <= 8; ++p) {
    for (int t = 1; t <= p; ++t) {
      INFO("degree " << p << ", decrement " << t);
      check_matches_exact(pb::interpolating_reduction_operator(p, t), exact_operator(p, t, true));
    }
  }
}

TEST_CASE("reduce operators: the exact assembly holds at degrees 12, 16 and 20") {
  // The reason the operators are assembled in rationals: the interior Gram block grows
  // ill-conditioned with the degree, so a double solve would lose digits that the oracle (the same
  // definition, solved exactly by other code) keeps. Spot checks above degree 8, where the sweeps
  // above stop. Bug caught: an intermediate in double (a Gram entry or an elimination step) that
  // only bites at high degree.
  struct Pair {
    int degree;
    int decrement;
  };
  for (const Pair pr :
       {Pair{12, 1}, Pair{12, 5}, Pair{16, 3}, Pair{20, 1}, Pair{20, 2}, Pair{20, 11}}) {
    INFO("degree " << pr.degree << ", decrement " << pr.decrement);
    check_matches_exact(pb::interpolating_reduction_operator(pr.degree, pr.decrement),
                        exact_operator(pr.degree, pr.decrement, true));
    check_matches_exact(pb::l2_reduction_operator(pr.degree, pr.decrement),
                        exact_operator(pr.degree, pr.decrement, false));
  }
}

TEST_CASE("reduce operators: the L2 operator equals the exact-rational oracle") {
  for (int p = 1; p <= 8; ++p) {
    for (int t = 1; t <= p; ++t) {
      INFO("degree " << p << ", decrement " << t);
      check_matches_exact(pb::l2_reduction_operator(p, t), exact_operator(p, t, false));
    }
  }
}

TEST_CASE("reduce operators: the elevation matrix equals repeated unit elevation, exactly") {
  // Oracle from a different route: elevating by one repeats c'_i = (i / (m + 1)) c_{i-1} + (1 - i /
  // (m
  // + 1)) c_i, applied to the identity; the library uses the closed form C(q, j) C(t, i - j) / C(p,
  // i).
  for (int q = 0; q <= 8; ++q) {
    for (int t = 1; t <= 8; ++t) {
      Matrix e(static_cast<std::size_t>(q) + 1,
               std::vector<Rational>(static_cast<std::size_t>(q) + 1));
      for (int i = 0; i <= q; ++i) {
        e[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)] = 1;
      }
      for (int m = q; m < q + t; ++m) {
        Matrix next(static_cast<std::size_t>(m) + 2,
                    std::vector<Rational>(static_cast<std::size_t>(q) + 1, Rational(0)));
        for (int i = 0; i <= m + 1; ++i) {
          for (int j = 0; j <= q; ++j) {
            Rational v = 0;
            if (i >= 1) {
              v += rat(i, m + 1) * e[static_cast<std::size_t>(i) - 1][static_cast<std::size_t>(j)];
            }
            if (i <= m) {
              v += rat(m + 1 - i, m + 1) *
                   e[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
            }
            next[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = v;
          }
        }
        e = std::move(next);
      }
      INFO("degree " << q << ", increment " << t);
      check_matches_exact(pb::elevation_matrix(q, t), e);
    }
  }
}

TEST_CASE("reduce operators: the cubic-to-quadratic interpolating row is (-1/4, 3/4, 3/4, -1/4)") {
  // Hand derivation (a closed form, independent of the oracle above): with q_0 = c_0 and q_2 = c_3
  // the elevation of q is Q' = (c_0, (c_0 + 2 q_1) / 3, (2 q_1 + c_3) / 3, c_3); the difference
  // d = c - Q' has d_1 = a - 2 q_1 / 3, d_2 = b - 2 q_1 / 3 with a = c_1 - c_0 / 3, b = c_2 - c_3
  // / 3. The cubic Gram matrix has G_11 = G_22 = 3/35 and G_12 = 9/140, so d/dq_1 (d^T G d) =
  // -(2/3) (G_11 + G_12) (d_1 + d_2) vanishes at d_1 + d_2 = 0, i.e. q_1 = 3 (a + b) / 4. Every
  // entry is a dyadic rational, so the double operator holds it exactly.
  const NdArray<double> r = pb::interpolating_reduction_operator(3, 1);
  REQUIRE(r.shape()[0] == 3);
  REQUIRE(r.shape()[1] == 4);
  const double want[12] = {1, 0, 0, 0, -0.25, 0.75, 0.75, -0.25, 0, 0, 0, 1};
  for (std::size_t i = 0; i < 12; ++i) {
    CHECK(r.data()[i] == want[i]);
  }
}

TEST_CASE("reduce operators: a reduction to degree 0 is the mean, a row of 1/(p + 1)") {
  // The best constant for f = sum c_j B_j is its mean, and the integral of each B_j is 1/(p + 1).
  // The interpolating variant cannot keep two end values at degree 0 and returns the same.
  for (int p = 1; p <= 12; ++p) {
    for (const bool interpolating : {true, false}) {
      const NdArray<double> r = interpolating ? pb::interpolating_reduction_operator(p, p)
                                              : pb::l2_reduction_operator(p, p);
      REQUIRE(r.shape()[0] == 1);
      REQUIRE(r.shape()[1] == static_cast<std::size_t>(p) + 1);
      const double want = 1.0 / static_cast<double>(p + 1);  // correctly rounded, like the entry
      for (std::size_t j = 0; j < r.shape()[1]; ++j) {
        CAPTURE(p);
        CHECK(std::fabs(r.data()[j] - want) <= 2.0 * u_double * want);
      }
    }
  }
}

TEST_CASE("reduce operators: the interpolating rows keep the ends and reproduce constants") {
  // Invariants. First and last rows are unit vectors (so end values pass through unrounded), and
  // every row sums to 1 (a constant polynomial is reproduced), up to the entry rounding: the sum of
  // a row of an operator with Lebesgue sum L is within 6 u L of 1 plus the summation rounding
  // (p + 1) u L.
  for (int p = 2; p <= 12; ++p) {
    for (int t = 1; t < p; ++t) {
      const NdArray<double> r = pb::interpolating_reduction_operator(p, t);
      const std::size_t rows = r.shape()[0];
      const std::size_t cols = r.shape()[1];
      INFO("degree " << p << ", decrement " << t);
      for (std::size_t j = 0; j < cols; ++j) {
        CHECK(r.data()[j] == (j == 0 ? 1.0 : 0.0));
        CHECK(r.data()[(rows - 1) * cols + j] == (j == cols - 1 ? 1.0 : 0.0));
      }
      for (std::size_t i = 0; i < rows; ++i) {
        double sum = 0.0;
        double abs_sum = 0.0;
        for (std::size_t j = 0; j < cols; ++j) {
          sum += r.data()[i * cols + j];
          abs_sum += std::fabs(r.data()[i * cols + j]);
        }
        CAPTURE(i);
        CHECK(std::fabs(sum - 1.0) <= (6.0 + static_cast<double>(cols)) * u_double * abs_sum);
      }
    }
  }
}

// ---- reduce_degree
// -------------------------------------------------------------------------------

TEST_CASE_TEMPLATE("reduce_degree: the end values are reproduced bit for bit", T, float, double) {
  // Bug caught: an operator whose first or last row is rounded or contracted into an fma, so that a
  // reduced curve no longer starts and ends where the original does (and neighbouring segments
  // of a spline stop meeting).
  Lcg rng;
  for (std::size_t rank = 1; rank <= 3; ++rank) {
    for (std::size_t p = 2; p <= 9; ++p) {
      for (long long t = 1; t < static_cast<long long>(p); ++t) {
        const std::vector<T> c = random_vec<T>(rng, (p + 1) * rank);
        const pb::Bezier<T> b = make_bezier<T>(c, {p + 1, rank});
        const pb::Bezier<T> r = pb::reduce_degree<T>(b, std::vector<long long>{t});
        const std::size_t q = p - static_cast<std::size_t>(t);
        INFO("rank " << rank << ", degree " << p << ", decrement " << t);
        REQUIRE(r.degree(0) == q);
        CHECK_FALSE(r.rational());
        CHECK(r.cp_size() == rank);
        for (std::size_t k = 0; k < rank; ++k) {
          CHECK(same_bits(r.control_points()[k], c[k]));
          CHECK(same_bits(r.control_points()[q * rank + k], c[p * rank + k]));
        }
      }
    }
  }
}

TEST_CASE_TEMPLATE("reduce_degree: the corners of a reduced surface are the original corners", T,
                   float, double) {
  Lcg rng;
  const std::size_t n0 = 5;
  const std::size_t n1 = 4;
  const std::size_t rank = 2;
  const std::vector<T> c = random_vec<T>(rng, n0 * n1 * rank);
  const pb::Bezier<T> b = make_bezier<T>(c, {n0, n1, rank});
  const pb::Bezier<T> r = pb::reduce_degree<T>(b, std::vector<long long>{2, 1});
  REQUIRE(r.dim() == 2);
  REQUIRE(r.degree(0) == 2);
  REQUIRE(r.degree(1) == 2);
  const auto at = [&](const std::vector<T>& net, std::size_t cols, std::size_t i, std::size_t j,
                      std::size_t k) { return net[(i * cols + j) * rank + k]; };
  const std::vector<T> rc = copy_of(r);
  for (std::size_t k = 0; k < rank; ++k) {
    CHECK(same_bits(at(rc, 3, 0, 0, k), at(c, n1, 0, 0, k)));
    CHECK(same_bits(at(rc, 3, 0, 2, k), at(c, n1, 0, n1 - 1, k)));
    CHECK(same_bits(at(rc, 3, 2, 0, k), at(c, n1, n0 - 1, 0, k)));
    CHECK(same_bits(at(rc, 3, 2, 2, k), at(c, n1, n0 - 1, n1 - 1, k)));
  }
}

TEST_CASE_TEMPLATE("reduce_degree: a surface is reduced by the oracle operator along each axis", T,
                   float, double) {
  // Bug caught: the operator applied along the wrong axis, or to the wrong stride, of a net with
  // more than one parametric direction and more than one component (flatten and unflatten).
  // Oracle: the exact operators applied in exact arithmetic to the exact values of the stored data.
  // Per output entry the error is the entry rounding (6 u_double), the double accumulation (p + 1
  // terms) and the one rounding to T, all times the sum of |R_ij c_j|: (p + 8) u_T of it.
  Lcg rng;
  const std::size_t n0 = 5;
  const std::size_t n1 = 4;
  const std::size_t rank = 2;
  const std::vector<T> c = random_vec<T>(rng, n0 * n1 * rank);
  const pb::Bezier<T> b = make_bezier<T>(c, {n0, n1, rank});
  const double u = unit_roundoff<T>();
  struct Case {
    long long d0;
    long long d1;
  };
  for (const Case cs : {Case{1, 0}, Case{0, 2}, Case{2, 1}}) {
    INFO("decrements " << cs.d0 << ", " << cs.d1);
    const pb::Bezier<T> r = pb::reduce_degree<T>(b, std::vector<long long>{cs.d0, cs.d1});
    const std::size_t m0 = n0 - static_cast<std::size_t>(cs.d0);
    const std::size_t m1 = n1 - static_cast<std::size_t>(cs.d1);
    REQUIRE(r.degree(0) == m0 - 1);
    REQUIRE(r.degree(1) == m1 - 1);
    // Exact: reduce axis 0, then axis 1, in rationals; the abs-sum carries both stages.
    std::vector<Rational> exact(c.size());
    std::vector<Rational> mag(c.size());
    for (std::size_t i = 0; i < c.size(); ++i) {
      exact[i] = Rational(static_cast<double>(c[i]));
      mag[i] = abs_of(exact[i]);
    }
    std::size_t s0 = n0;
    std::size_t s1 = n1;
    if (cs.d0 > 0) {
      const Matrix op = exact_operator(static_cast<int>(n0) - 1, static_cast<int>(cs.d0), true);
      std::vector<Rational> next(m0 * s1 * rank, Rational(0));
      std::vector<Rational> next_mag(next.size(), Rational(0));
      for (std::size_t i = 0; i < m0; ++i) {
        for (std::size_t j = 0; j < s1; ++j) {
          for (std::size_t k = 0; k < rank; ++k) {
            for (std::size_t a = 0; a < s0; ++a) {
              next[(i * s1 + j) * rank + k] += op[i][a] * exact[(a * s1 + j) * rank + k];
              next_mag[(i * s1 + j) * rank + k] += abs_of(op[i][a]) * mag[(a * s1 + j) * rank + k];
            }
          }
        }
      }
      exact = std::move(next);
      mag = std::move(next_mag);
      s0 = m0;
    }
    if (cs.d1 > 0) {
      const Matrix op = exact_operator(static_cast<int>(n1) - 1, static_cast<int>(cs.d1), true);
      std::vector<Rational> next(s0 * m1 * rank, Rational(0));
      std::vector<Rational> next_mag(next.size(), Rational(0));
      for (std::size_t i = 0; i < s0; ++i) {
        for (std::size_t j = 0; j < m1; ++j) {
          for (std::size_t k = 0; k < rank; ++k) {
            for (std::size_t a = 0; a < s1; ++a) {
              next[(i * m1 + j) * rank + k] += op[j][a] * exact[(i * s1 + a) * rank + k];
              next_mag[(i * m1 + j) * rank + k] += abs_of(op[j][a]) * mag[(i * s1 + a) * rank + k];
            }
          }
        }
      }
      exact = std::move(next);
      mag = std::move(next_mag);
    }
    const std::span<const T> got = r.control_points();
    REQUIRE(got.size() == exact.size());
    // Two stages double the rounding terms.
    const double stages = (cs.d0 > 0 ? 1.0 : 0.0) + (cs.d1 > 0 ? 1.0 : 0.0);
    for (std::size_t i = 0; i < got.size(); ++i) {
      const double bound = stages * (static_cast<double>(n0 + n1) + 8.0) * u * to_double(mag[i]);
      CAPTURE(i);
      CHECK(std::fabs(static_cast<double>(got[i]) - to_double(exact[i])) <= bound);
    }
  }
}

TEST_CASE_TEMPLATE("reduce_degree: reducing an elevated curve returns it (R E = I)", T, float,
                   double) {
  // Round trip. The elevated coefficients carry (p + 4) u B of rounding each, the operator has
  // Lebesgue sum L (the largest absolute row sum, from the exact oracle), and applying it adds the
  // entry rounding and the accumulation: |error| <= L (2 p + 7) u B. degree_reduction_error of the
  // same curve is the Gram norm (at most the Euclidean norm of the coefficients) of E R c - c,
  // which is (E R - I) of the elevation rounding: at most sqrt(p + 1) (L + 1) (2 p + 7) u B.
  Lcg rng;
  const double u = unit_roundoff<T>();
  for (std::size_t q = 1; q <= 5; ++q) {
    for (long long t = 1; t <= 4; ++t) {
      const std::size_t p = q + static_cast<std::size_t>(t);
      const std::size_t rank = 2;
      const std::vector<T> low = random_vec<T>(rng, (q + 1) * rank);
      const pb::Bezier<T> original = make_bezier<T>(low, {q + 1, rank});
      const pb::Bezier<T> high = pb::elevate_degree<T>(original, std::vector<long long>{t});
      const pb::Bezier<T> back = pb::reduce_degree<T>(high, std::vector<long long>{t});
      const double lebesgue =
          lebesgue_of(exact_operator(static_cast<int>(p), static_cast<int>(t), true));
      const double big = largest_magnitude(low);
      INFO("degree " << q << " elevated by " << t);
      REQUIRE(back.degree(0) == q);
      const std::vector<T> got = copy_of(back);
      for (std::size_t i = 0; i < got.size(); ++i) {
        CAPTURE(i);
        CHECK(std::fabs(static_cast<double>(got[i]) - static_cast<double>(low[i])) <=
              lebesgue * (2.0 * static_cast<double>(p) + 7.0) * u * big);
      }
      const double err = pb::degree_reduction_error<T>(high, std::vector<long long>{t});
      CHECK(err >= 0.0);
      CHECK(err <= std::sqrt(static_cast<double>(p) + 1.0) * (lebesgue + 1.0) *
                       (2.0 * static_cast<double>(p) + 7.0) * u * big);
    }
  }
}

TEST_CASE("degree_reduction_error: t^3, t^2 and t have the hand-integrated L2 errors") {
  // Analytic. (i) t^3 = (0, 0, 0, 1) to a quadratic: the reduction is -t/2 + 3t^2/2, the difference
  // t (t - 1/2) (t - 1) = u^3 - u/4 (u = t - 1/2), and the integral of its square over u in
  // [-1/2, 1/2] is 2 (2^-7/7 - 2^-5/10 + 2^-3/48) = 1/840. (ii) t^2 = (0, 0, 1) to a line: the
  // reduction is t, the difference -t (1 - t) and the integral of its square is 1/30. (iii) t =
  // (0, 1) to a constant: the mean 1/2, difference t - 1/2, integral of the square 1/12.
  // Tolerance: the coefficient difference has rounding of at most 7 u each (B = 1), its Gram norm
  // is at most its Euclidean norm (Gram eigenvalues are at most 1/(p + 1)), 2 for four entries; 32
  // (p + 4) u is generous by a factor 16 over that.
  const double tol3 = 32.0 * 7.0 * u_double;
  const std::vector<double> cubic = {0.0, 0.0, 0.0, 1.0};
  const pb::Bezier<double> t3 = make_bezier<double>(cubic, {4, 1});
  CHECK(std::fabs(pb::degree_reduction_error<double>(t3, std::vector<long long>{1}) -
                  1.0 / std::sqrt(840.0)) <= tol3);
  const std::vector<double> quad = {0.0, 0.0, 1.0};
  const pb::Bezier<double> t2 = make_bezier<double>(quad, {3, 1});
  CHECK(std::fabs(pb::degree_reduction_error<double>(t2, std::vector<long long>{1}) -
                  1.0 / std::sqrt(30.0)) <= tol3);
  const std::vector<double> line = {0.0, 1.0};
  const pb::Bezier<double> t1 = make_bezier<double>(line, {2, 1});
  CHECK(std::fabs(pb::degree_reduction_error<double>(t1, std::vector<long long>{1}) -
                  1.0 / std::sqrt(12.0)) <= tol3);
}

// ---- Rational nets -----------------------------------------------------------------------------

TEST_CASE("reduce_degree: a reduction that inverts a weight is refused, not returned") {
  // Weights (1, 0.1, 0.12, 1) are positive, but the interpolating cubic-to-quadratic weight row is
  // (-1/4, 3/4, 3/4, -1/4): the reduced middle weight is (-1 + 0.3 + 0.36 - 1) / 4 = -0.335 < 0, so
  // the reduced denominator can vanish. Bug caught: a rational curve returned with a pole inside
  // [0, 1]. (The weights are not an exact quadratic's elevation, so the round-trip error is
  // positive.)
  const std::vector<double> homogeneous = {0.0, 1.0, 0.1, 0.1, 0.2, 0.12, 0.3, 1.0};
  const pb::Bezier<double> curve = make_bezier<double>(homogeneous, {4, 2}, true);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(pb::reduce_degree<double>(curve, std::vector<long long>{1})),
      doctest::Contains("not strictly positive"), pantr::value_error);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(pb::reduce_degree<double>(curve, std::vector<long long>{1})),
      doctest::Contains("-3.350e-01"), pantr::value_error);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(pb::reduce_degree<double>(curve, std::vector<long long>{1})),
      doctest::Contains("Use minimize_degree"), pantr::value_error);
  // The error of the refused reduction is still reported, as the previous release did.
  CHECK(pb::degree_reduction_error<double>(curve, std::vector<long long>{1}) > 0.0);
  // The same operator on a non-rational net of the same numbers is not checked at all.
  const pb::Bezier<double> plain = make_bezier<double>(homogeneous, {4, 2}, false);
  CHECK_NOTHROW(static_cast<void>(pb::reduce_degree<double>(plain, std::vector<long long>{1})));
  // A NaN weight is refused too (it would compare false against every bound).
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const std::vector<double> with_nan = {0.0, 1.0, 0.1, 1.0, 0.2, nan, 0.3, 1.0};
  const pb::Bezier<double> nan_curve = make_bezier<double>(with_nan, {4, 2}, true);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(pb::reduce_degree<double>(nan_curve, std::vector<long long>{1})),
      doctest::Contains("not strictly positive"), pantr::value_error);
  // Equal positive weights reduce to equal positive weights: no refusal.
  const std::vector<double> flat = {0.0, 1.0, 0.1, 1.0, 0.2, 1.0, 0.3, 1.0};
  const pb::Bezier<double> ok = make_bezier<double>(flat, {4, 2}, true);
  const pb::Bezier<double> reduced = pb::reduce_degree<double>(ok, std::vector<long long>{1});
  CHECK(reduced.rational());
  CHECK(reduced.degree(0) == 2);
}

// ---- Decrement validation ----------------------------------------------------------------------

TEST_CASE("reduce_degree: invalid decrements are refused with the documented messages") {
  const std::vector<double> c(4, 1.0);
  const pb::Bezier<double> curve = make_bezier<double>(c, {4, 1});
  const std::vector<double> s(12, 1.0);
  const pb::Bezier<double> surf = make_bezier<double>(s, {4, 3, 1});
  CHECK_THROWS_WITH_AS(
      static_cast<void>(pb::reduce_degree<double>(curve, std::vector<long long>{1, 1})),
      "Number of degree decrements (2) must match dimension (1).", pantr::value_error);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(pb::reduce_degree<double>(curve, std::vector<long long>{-1})),
      "Degree decrements must be non-negative.", pantr::value_error);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(pb::reduce_degree<double>(curve, std::vector<long long>{0})),
      "At least one degree decrement must be positive.", pantr::value_error);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(pb::reduce_degree<double>(curve, std::vector<long long>{4})),
      "Degree decrement (4) in direction 0 exceeds current degree (3).", pantr::value_error);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(pb::reduce_degree<double>(surf, std::vector<long long>{1, 3})),
      "Degree decrement (3) in direction 1 exceeds current degree (2).", pantr::value_error);
  // The negative entry is reported before "no positive entry" and before the range of the others.
  CHECK_THROWS_WITH_AS(
      static_cast<void>(pb::reduce_degree<double>(surf, std::vector<long long>{0, -1})),
      "Degree decrements must be non-negative.", pantr::value_error);
  // degree_reduction_error validates alike (the shared check).
  CHECK_THROWS_WITH_AS(
      static_cast<void>(pb::degree_reduction_error<double>(curve, std::vector<long long>{4})),
      "Degree decrement (4) in direction 0 exceeds current degree (3).", pantr::value_error);
  // A decrement equal to the degree is allowed (the mean), and the boundary one below it too.
  CHECK_NOTHROW(static_cast<void>(pb::reduce_degree<double>(curve, std::vector<long long>{3})));
  CHECK_NOTHROW(static_cast<void>(pb::reduce_degree<double>(curve, std::vector<long long>{2})));
}

// ---- minimize_degree ---------------------------------------------------------------------------

TEST_CASE_TEMPLATE("minimize_degree: an elevated polynomial curve returns to its base degree", T,
                   float, double) {
  // Bug caught: a tolerance test that never accepts (or always does), a norm that is not the L^2
  // norm of the elevated difference, a round trip that stops one degree early. The base curve is
  // generic, so degree q is minimal; the elevated one has degree q + 3. Closeness: each accepted
  // step has relative L^2 round-trip error below tol = 1e3 eps, so the result elevated back is
  // within 3 * tol * (1 + 3 tol) of the original in the relative L^2 norm (triangle inequality).
  Lcg rng;
  const double tol = 1.0e3 * static_cast<double>(std::numeric_limits<T>::epsilon());
  for (std::size_t q = 1; q <= 4; ++q) {
    const std::size_t rank = 3;
    const std::vector<T> low = random_vec<T>(rng, (q + 1) * rank);
    const pb::Bezier<T> base = make_bezier<T>(low, {q + 1, rank});
    const pb::Bezier<T> high = pb::elevate_degree<T>(base, std::vector<long long>{3});
    const pb::Bezier<T> min = pb::minimize_degree<T>(high, std::nullopt);
    INFO("base degree " << q);
    REQUIRE(min.degree(0) == q);
    CHECK_FALSE(min.rational());
    CHECK_FALSE(min.same_rep(high));
    const pb::Bezier<T> again = pb::elevate_degree<T>(min, std::vector<long long>{3});
    std::vector<double> diff(high.size());
    std::vector<double> ref(high.size());
    for (std::size_t i = 0; i < diff.size(); ++i) {
      diff[i] = static_cast<double>(again.control_points()[i]) -
                static_cast<double>(high.control_points()[i]);
      ref[i] = static_cast<double>(high.control_points()[i]);
    }
    const std::vector<std::size_t> orders = {q + 4};
    CHECK(exact_l2_norm(diff, orders, rank) <=
          3.0 * tol * (1.0 + 3.0 * tol) * exact_l2_norm(ref, orders, rank));
  }
}

TEST_CASE("minimize_degree: a surface returns to its base degree in each direction") {
  Lcg rng;
  const std::vector<double> low = random_vec<double>(rng, 3 * 2 * 2);
  const pb::Bezier<double> base = make_bezier<double>(low, {3, 2, 2});  // degrees (2, 1)
  const pb::Bezier<double> high = pb::elevate_degree<double>(base, std::vector<long long>{2, 3});
  REQUIRE(high.degree(0) == 4);
  REQUIRE(high.degree(1) == 4);
  const pb::Bezier<double> min = pb::minimize_degree<double>(high, std::nullopt);
  CHECK(min.degree(0) == 2);
  CHECK(min.degree(1) == 1);
  CHECK(min.cp_size() == 2);
}

TEST_CASE("minimize_degree: an elevated rational curve returns to its base degree") {
  // The rational quarter circle x^2 + y^2 = 1: control points (1, 0), (1, 1), (0, 1) with weights
  // (1, sqrt(2)/2, 1), stored homogeneously as (x w, y w, w). Elevated by 2, then minimised: the
  // rational round trip is measured on the projected map and the exactly reducible curve reaches
  // degree 2, not lower (a quarter circle is not a rational line).
  const double h = std::sqrt(2.0) / 2.0;
  const std::vector<double> circle = {1.0, 0.0, 1.0, h, h, h, 0.0, 1.0, 1.0};
  const pb::Bezier<double> base = make_bezier<double>(circle, {3, 3}, true);
  const pb::Bezier<double> high = pb::elevate_degree<double>(base, std::vector<long long>{2});
  REQUIRE(high.degree(0) == 4);
  const pb::Bezier<double> min = pb::minimize_degree<double>(high, std::nullopt);
  CHECK(min.rational());
  CHECK(min.degree(0) == 2);
  CHECK(min.rank() == 2);
}

TEST_CASE("minimize_degree: the tolerance is a strict relative L2 threshold") {
  // Analytic. For t^3 = (0, 0, 0, 1): the round-trip error to a quadratic is 1/sqrt(840) and
  // ||t^3|| = 1/sqrt(7), so the relative error is sqrt(7 / 840) = sqrt(1/120) = 0.091287...; the
  // trial is rejected at tol <= that (the test is `rel >= tol`) and accepted above. Reducing the
  // quadratic -t/2 + 3t^2/2 = (0, -1/4, 1) further to a line has relative error 0.688 (round-trip
  // difference (0, -3/4, 0) of norm sqrt(3/40) against the norm sqrt(19/120) of the quadratic), so
  // 0.0914 stops at degree 2. Margins of 0.1% on both sides are far above the rounding (1e-14).
  const std::vector<double> cubic = {0.0, 0.0, 0.0, 1.0};
  const pb::Bezier<double> t3 = make_bezier<double>(cubic, {4, 1});
  const pb::Bezier<double> kept = pb::minimize_degree<double>(t3, 0.0912);
  CHECK(kept.degree(0) == 3);
  const pb::Bezier<double> reduced = pb::minimize_degree<double>(t3, 0.0914);
  REQUIRE(reduced.degree(0) == 2);
  const double want[3] = {0.0, -0.25, 1.0};
  for (std::size_t i = 0; i < 3; ++i) {
    CHECK(std::fabs(reduced.control_points()[i] - want[i]) <= 64.0 * u_double);
  }
}

TEST_CASE_TEMPLATE("minimize_degree: nothing to reduce gives a copy, and tol <= 0 keeps the degree",
                   T, float, double) {
  // The result never shares the input's buffer, even unchanged (the input may be written by
  // someone else through the buffer constructor).
  const std::vector<T> cubic = {T(0), T(0), T(0), T(1)};
  const pb::Bezier<T> t3 = make_bezier<T>(cubic, {4, 1});
  const pb::Bezier<T> same = pb::minimize_degree<T>(t3, 1e-3);  // 0.0913 > 1e-3: kept
  CHECK(same.degree(0) == 3);
  CHECK_FALSE(same.same_rep(t3));
  CHECK(same.control_points().data() != t3.control_points().data());
  for (std::size_t i = 0; i < 4; ++i) {
    CHECK(same.control_points()[i] == t3.control_points()[i]);
  }
  // An exactly reducible curve is kept at degree 3 when tol <= 0, whatever the sign.
  const std::vector<T> line = {T(0), T(1) / 3, T(2) / 3, T(1)};
  const pb::Bezier<T> elevated_line = make_bezier<T>(line, {4, 1});
  for (const double tol : {0.0, -1.0, -1e-12}) {
    const pb::Bezier<T> kept = pb::minimize_degree<T>(elevated_line, tol);
    CAPTURE(tol);
    CHECK(kept.degree(0) == 3);
    CHECK_FALSE(kept.same_rep(elevated_line));
  }
  CHECK(pb::minimize_degree<T>(elevated_line, std::nullopt).degree(0) == 1);
}

TEST_CASE("minimize_degree: past degree 61 the binomial envelope is refused, at 61 it is not") {
  // Bug caught: the exact-integer binomial kernel wrapping silently past 61, so a degree-62 curve
  // is "reduced" with corrupted coefficients. tol = 0 keeps the degree and does no arithmetic, so
  // the envelope check, which precedes it, can be exercised at the boundary cheaply.
  const std::vector<double> c61(62, 1.0);
  const std::vector<double> c62(63, 1.0);
  const pb::Bezier<double> deg61 = make_bezier<double>(c61, {62, 1});
  const pb::Bezier<double> deg62 = make_bezier<double>(c62, {63, 1});
  CHECK_NOTHROW(static_cast<void>(pb::minimize_degree<double>(deg61, 0.0)));
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::minimize_degree<double>(deg62, 0.0)),
                       doctest::Contains("Degree minimization of a degree-62 direction 0"),
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::minimize_degree<double>(deg62, std::nullopt)),
                       doctest::Contains("degree-62 direction 0"), pantr::value_error);
  // The offending direction is named.
  const std::vector<double> s(3 * 63, 1.0);
  const pb::Bezier<double> surf = make_bezier<double>(s, {3, 63, 1});
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::minimize_degree<double>(surf, 0.0)),
                       doctest::Contains("degree-62 direction 1"), pantr::value_error);
}

// ---- The process table -------------------------------------------------------------------------

TEST_CASE("reduction table: a key yields one shared operator, equal to the direct computation") {
  pb::reduction_cache_clear();
  CHECK(pb::reduction_cache_size() == 0);
  const auto a = pb::cached_reduction_operator(pb::ReductionKind::interpolating, 5, 2);
  const auto b = pb::cached_reduction_operator(pb::ReductionKind::interpolating, 5, 2);
  CHECK(a.get() == b.get());  // the same immutable matrix, not an equal copy
  CHECK(pb::reduction_cache_size() == 1);
  const NdArray<double> direct = pb::interpolating_reduction_operator(5, 2);
  REQUIRE(a->size() == direct.size());
  for (std::size_t i = 0; i < direct.size(); ++i) {
    CHECK(a->data()[i] == direct.data()[i]);
  }
  // Another kind, another degree, another decrement: each its own entry.
  const auto l2 = pb::cached_reduction_operator(pb::ReductionKind::l2, 5, 2);
  const auto other_degree = pb::cached_reduction_operator(pb::ReductionKind::interpolating, 6, 2);
  const auto other_decrement =
      pb::cached_reduction_operator(pb::ReductionKind::interpolating, 5, 3);
  CHECK(l2.get() != a.get());
  CHECK(other_degree.get() != a.get());
  CHECK(other_decrement.get() != a.get());
  CHECK(pb::reduction_cache_size() == 4);
  // The L2 entry is the L2 operator, not the interpolating one filed under another key.
  const NdArray<double> direct_l2 = pb::l2_reduction_operator(5, 2);
  for (std::size_t i = 0; i < direct_l2.size(); ++i) {
    CHECK(l2->data()[i] == direct_l2.data()[i]);
  }
  CHECK_FALSE(l2->data()[0] == a->data()[0]);  // 1 for the interpolating row, the L2 one is not
}

TEST_CASE("reduction table: clearing empties it and leaves handed-out operators valid") {
  const auto before = pb::cached_reduction_operator(pb::ReductionKind::interpolating, 4, 1);
  const std::vector<double> snapshot(before->data(), before->data() + before->size());
  pb::reduction_cache_clear();
  CHECK(pb::reduction_cache_size() == 0);
  for (std::size_t i = 0; i < snapshot.size(); ++i) {
    CHECK(before->data()[i] == snapshot[i]);
  }
  const auto after = pb::cached_reduction_operator(pb::ReductionKind::interpolating, 4, 1);
  CHECK(after.get() != before.get());  // recomputed after the clear
  CHECK(pb::reduction_cache_size() == 1);
  for (std::size_t i = 0; i < snapshot.size(); ++i) {
    CHECK(after->data()[i] == snapshot[i]);
  }
}

TEST_CASE("reduction table: out-of-range pairs are refused and degrees past the cap are not kept") {
  CHECK_THROWS_AS(static_cast<void>(pb::cached_reduction_operator(pb::ReductionKind::l2, 0, 1)),
                  pantr::value_error);
  CHECK_THROWS_AS(static_cast<void>(pb::cached_reduction_operator(pb::ReductionKind::l2, 3, 0)),
                  pantr::value_error);
  CHECK_THROWS_AS(
      static_cast<void>(pb::cached_reduction_operator(pb::ReductionKind::interpolating, 3, 4)),
      pantr::value_error);
  CHECK_THROWS_AS(static_cast<void>(pb::interpolating_reduction_operator(0, 1)),
                  pantr::value_error);
  CHECK_THROWS_AS(static_cast<void>(pb::l2_reduction_operator(3, 4)), pantr::value_error);
  CHECK_THROWS_AS(static_cast<void>(pb::elevation_matrix(-1, 1)), pantr::value_error);
  CHECK_THROWS_AS(static_cast<void>(pb::elevation_matrix(2, 0)), pantr::value_error);
  pb::reduction_cache_clear();
  // One degree past the cap: each call computes a fresh operator and none is kept. (Reduced to a
  // line, so the exact solve is trivial.)
  const long long past = pb::reduction_cache_max_degree + 1;
  const auto a = pb::cached_reduction_operator(pb::ReductionKind::interpolating, past, past - 1);
  const auto b = pb::cached_reduction_operator(pb::ReductionKind::interpolating, past, past - 1);
  CHECK(a.get() != b.get());
  CHECK(pb::reduction_cache_size() == 0);
  REQUIRE(a->size() == b->size());
  for (std::size_t i = 0; i < a->size(); ++i) {
    CHECK(a->data()[i] == b->data()[i]);
  }
  // At the cap itself it is kept.
  const long long cap = pb::reduction_cache_max_degree;
  const auto c = pb::cached_reduction_operator(pb::ReductionKind::interpolating, cap, cap - 1);
  const auto d = pb::cached_reduction_operator(pb::ReductionKind::interpolating, cap, cap - 1);
  CHECK(c.get() == d.get());
  CHECK(pb::reduction_cache_size() == 1);
}

TEST_CASE("reduction table: concurrent first use of the same keys yields one operator each") {
  // The table's race (a TSan target): threads that find a key absent each compute it outside the
  // lock, and the first stored wins. Bug caught: a lookup, a size query or an insertion outside the
  // mutex (a data race on the map), or an insertion that replaces, so that two threads hold
  // different operators for one key. Eight writers visit the keys from different starting keys.
  // Two pollers that never take the table's mutex themselves read it all the while, one by size
  // and one by looking up a key: a reader that did acquire the mutex between its accesses would
  // order them after the writers' and hide a missing lock from the sanitizer. The values the
  // pollers read are used, since a discarded read would be optimised away.
  pb::reduction_cache_clear();
  constexpr std::size_t n_threads = 8;
  constexpr std::size_t n_keys = 4;
  const long long decrements[n_keys] = {1, 2, 3, 4};
  // Key 0 is entered before the threads start, so that the lookup poller always hits and never
  // takes the mutex for an insertion of its own (which would order it after the writers).
  const auto warm = pb::cached_reduction_operator(pb::ReductionKind::interpolating, 14, 1);
  std::vector<std::vector<std::shared_ptr<const NdArray<double>>>> got(
      n_threads, std::vector<std::shared_ptr<const NdArray<double>>>(n_keys));
  std::vector<std::thread> threads;
  std::atomic<std::size_t> ready{0};
  std::atomic<bool> go{false};
  std::atomic<bool> stop{false};
  std::atomic<std::size_t> largest_size{0};
  std::atomic<std::size_t> polled_changes{0};
  std::shared_ptr<const NdArray<double>> polled_first;
  std::thread size_poller([&] {
    while (!stop.load()) {
      std::size_t seen = largest_size.load();
      const std::size_t size = pb::reduction_cache_size();
      while (size > seen && !largest_size.compare_exchange_weak(seen, size)) {
      }
      std::this_thread::yield();
    }
  });
  std::thread lookup_poller([&] {
    while (!stop.load()) {
      const auto op =
          pb::cached_reduction_operator(pb::ReductionKind::interpolating, 14, decrements[0]);
      if (!polled_first) {
        polled_first = op;
      } else if (polled_first.get() != op.get()) {
        polled_changes.fetch_add(1);  // an entry replaced after it had been handed out
      }
      std::this_thread::yield();
    }
  });
  for (std::size_t t = 0; t < n_threads; ++t) {
    threads.emplace_back([&, t] {
      ready.fetch_add(1);
      while (!go.load()) {
        std::this_thread::yield();
      }
      for (std::size_t j = 0; j < n_keys; ++j) {
        const std::size_t k = (t + j) % n_keys;
        got[t][k] =
            pb::cached_reduction_operator(pb::ReductionKind::interpolating, 14, decrements[k]);
      }
    });
  }
  while (ready.load() < n_threads) {
    std::this_thread::yield();
  }
  go.store(true);
  for (auto& th : threads) {
    th.join();
  }
  stop.store(true);
  size_poller.join();
  lookup_poller.join();
  for (std::size_t k = 0; k < n_keys; ++k) {
    std::set<const void*> distinct;
    for (std::size_t t = 0; t < n_threads; ++t) {
      distinct.insert(got[t][k].get());
    }
    CAPTURE(k);
    CHECK(distinct.size() == 1);
  }
  for (std::size_t t = 0; t < n_threads; ++t) {
    CHECK(got[t][0].get() == warm.get());  // the entry made before the race is never replaced
  }
  CHECK(polled_changes.load() == 0);
  CHECK(largest_size.load() <= n_keys);  // the table never holds more than the distinct keys
  CHECK(pb::reduction_cache_size() == n_keys);
}
