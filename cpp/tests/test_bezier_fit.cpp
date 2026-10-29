// Bezier fitting: the truncated-SVD pseudo-inverse of the Bernstein Vandermonde matrix, tensor-grid
// fitting and scattered fitting.
//
// Oracles. Analytic / encoded truth: Bernstein coefficients chosen as dyadic numbers (exact in
// float and double), sampled at the chosen nodes by exact rational evaluation of the textbook sum
// C(n, i) x^i (1 - x)^(n - i) (the nodes are doubles, converted exactly to rationals), then
// rounded once to the storage type; the fit must give the coefficients back. Closed forms: the
// mean (degree 0), the identity (degree 1 at the two end nodes) and a matrix with the known
// singular values (sqrt 2, 1) for the truncation. Invariant: the least-squares solution has a
// residual orthogonal to the columns of the Vandermonde matrix.
//
// Tolerances. The pseudo-inverse is computed by an SVD that is backward stable, so the fitted
// coefficients of data in the model space carry a forward error of at most K (m + n + 2) u kappa
// ||c|| in the matching norm, where kappa is the condition number of the Vandermonde matrix and K
// absorbs the explicit multiplication by the computed pseudo-inverse. kappa is obtained from the
// exact inverse (Gauss-Jordan in rationals): kappa_inf = ||V||_inf ||V^-1||_inf for a square
// matrix; for a tall one, kappa_2 <= sqrt(n kappa_inf(V^T V)) from the exact normal-equations
// inverse. K = 8 leaves a wide margin over the errors seen when the tests were written.
#include <doctest/doctest.h>

#include <pantr/bezier/bezier.hpp>
#include <pantr/bezier/fit.hpp>
#include <pantr/core/errors.hpp>
#include <pantr/core/exact_rational.hpp>
#include <pantr/core/mdspan.hpp>
#include <pantr/core/ndarray.hpp>
#include <pantr/quad/rules_1d.hpp>

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

namespace pb = pantr::bezier;
using pantr::NdArray;
using pantr::exact::Integer;
using pantr::exact::Rational;

namespace {

using Matrix = std::vector<std::vector<Rational>>;

constexpr double fit_slack = 8.0;  // K of the header comment

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

// B_i^n(x) = C(n, i) x^i (1 - x)^(n - i) for all i, in exact arithmetic.
std::vector<Rational> bernstein_values(int n, const Rational& x) {
  std::vector<Rational> out(static_cast<std::size_t>(n) + 1);
  for (int i = 0; i <= n; ++i) {
    Rational v(binomial(n, i));
    for (int k = 0; k < i; ++k) {
      v *= x;
    }
    for (int k = 0; k < n - i; ++k) {
      v *= (1 - x);
    }
    out[static_cast<std::size_t>(i)] = v;
  }
  return out;
}

Matrix solve(Matrix a, Matrix b) {
  const std::size_t n = a.size();
  for (std::size_t col = 0; col < n; ++col) {
    std::size_t piv = col;
    while (piv < n && a[piv][col] == 0) {
      ++piv;
    }
    REQUIRE(piv < n);  // a singular system is an error of the oracle's own inputs
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

Matrix identity(std::size_t n) {
  Matrix m(n, std::vector<Rational>(n, Rational(0)));
  for (std::size_t i = 0; i < n; ++i) {
    m[i][i] = 1;
  }
  return m;
}

double norm_inf(const Matrix& m) {
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

Matrix transposed(const Matrix& m) {
  Matrix t(m[0].size(), std::vector<Rational>(m.size()));
  for (std::size_t i = 0; i < m.size(); ++i) {
    for (std::size_t j = 0; j < m[0].size(); ++j) {
      t[j][i] = m[i][j];
    }
  }
  return t;
}

Matrix product(const Matrix& a, const Matrix& b) {
  Matrix c(a.size(), std::vector<Rational>(b[0].size(), Rational(0)));
  for (std::size_t i = 0; i < a.size(); ++i) {
    for (std::size_t k = 0; k < b.size(); ++k) {
      for (std::size_t j = 0; j < b[0].size(); ++j) {
        c[i][j] += a[i][k] * b[k][j];
      }
    }
  }
  return c;
}

// The exact Vandermonde matrix: row i is B_0..B_degree at nodes[i].
template <class T>
Matrix vandermonde(const std::vector<T>& nodes, int degree) {
  Matrix v;
  for (const T x : nodes) {
    v.push_back(bernstein_values(degree, Rational(static_cast<double>(x))));
  }
  return v;
}

// An upper bound of kappa_2 (square matrices: kappa_inf, which bounds it up to sqrt(n); tall ones:
// through the exact normal equations).
double condition_bound(const Matrix& v) {
  const std::size_t n = v[0].size();
  const Matrix vt = transposed(v);
  const Matrix normal = product(vt, v);
  const Matrix inv = solve(normal, identity(n));
  return std::sqrt(static_cast<double>(n) * norm_inf(normal) * norm_inf(inv));
}

// Dyadic coefficients, exact in float and double: (7 j mod 11 - 5) / 8 + shift.
double dyadic(std::size_t j, std::size_t salt) {
  return (static_cast<double>((7 * j + 3 * salt) % 11) - 5.0) / 8.0;
}

template <class T>
std::vector<T> nodes_of(long long n) {
  const NdArray<T> nd = pantr::quad::modified_chebyshev_nodes_1d<T>(n);
  return {nd.data(), nd.data() + n};
}

struct Lcg {
  std::uint64_t s = 424242;
  std::uint64_t next() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return s >> 33;
  }
};

}  // namespace

TEST_CASE("fit oracle: a double converts to a rational exactly") {
  // The oracle itself: the encoded values are exact only if Rational(double) is.
  CHECK(Rational(0.5) == rat(1, 2));
  CHECK(Rational(0.375) == rat(3, 8));
  CHECK(Rational(0.1) * 10 != rat(1));  // 0.1 is not 1/10: the conversion keeps the binary value
  CHECK(to_double(Rational(0.1)) == 0.1);
  const std::vector<Rational> b = bernstein_values(3, rat(1, 4));  // 27/64, 27/64, 9/64, 1/64
  CHECK(b[0] == rat(27, 64));
  CHECK(b[1] == rat(27, 64));
  CHECK(b[2] == rat(9, 64));
  CHECK(b[3] == rat(1, 64));
}

TEST_CASE_TEMPLATE("bernstein_pinv: exact interpolation reproduces a known polynomial", T, float,
                   double) {
  // Bug caught: a transposed or mis-scaled pseudo-inverse, a basis tabulated at the wrong nodes, a
  // truncation that drops a singular value that interpolation needs.
  const double u = unit_roundoff<T>();
  const int max_degree = std::is_same_v<T, float> ? 6 : 10;
  for (int n = 1; n <= max_degree; ++n) {
    const std::vector<T> nodes = nodes_of<T>(n + 1);
    const Matrix v = vandermonde<T>(nodes, n);
    const double kappa = condition_bound(v);
    // Truncation must not act: smallest singular value is >= sigma_max / kappa.
    REQUIRE(1.0 / kappa > 100.0 * static_cast<double>(std::numeric_limits<T>::epsilon()));
    std::vector<double> c(static_cast<std::size_t>(n) + 1);
    for (std::size_t j = 0; j < c.size(); ++j) {
      c[j] = dyadic(j, 1);
    }
    std::vector<T> y;
    for (const auto& row : v) {
      Rational s = 0;
      for (std::size_t j = 0; j < c.size(); ++j) {
        s += row[j] * Rational(c[j]);
      }
      y.push_back(static_cast<T>(to_double(s)));
    }
    const NdArray<T> pinv = pb::bernstein_pinv<T>(std::span<const T>(nodes), n, std::nullopt);
    REQUIRE(pinv.shape().size() == 2);
    REQUIRE(pinv.shape()[0] == static_cast<std::size_t>(n) + 1);
    REQUIRE(pinv.shape()[1] == static_cast<std::size_t>(n) + 1);
    const double bound = fit_slack * (2.0 * n + 2.0) * u * kappa * 1.0;  // ||c||_inf <= 5/8 < 1
    for (std::size_t j = 0; j < c.size(); ++j) {
      double s = 0.0;
      for (std::size_t i = 0; i < y.size(); ++i) {
        s += static_cast<double>(pinv.data()[j * y.size() + i]) * static_cast<double>(y[i]);
      }
      INFO("degree " << n << ", coefficient " << j << ", kappa bound " << kappa);
      CHECK(std::fabs(s - c[j]) <= bound);
    }
  }
}

TEST_CASE_TEMPLATE("bernstein_pinv: fewer coefficients than nodes is the least-squares fit", T,
                   float, double) {
  // Exact data of degree d sampled at m > d + 1 nodes is in the model space, so the fit returns
  // the degree-d coefficients (analytic). With perturbed data it is the least-squares solution,
  // whose residual is orthogonal to the columns: V^T (V c - y) = 0, evaluated in exact arithmetic
  // on the rounded fit. Bug caught: an interpolation of the first d + 1 nodes, or a fit that
  // ignores the surplus nodes.
  const double u = unit_roundoff<T>();
  const int m = 9;
  const std::vector<T> nodes = nodes_of<T>(m);
  for (int d = 0; d <= 4; ++d) {
    const Matrix v = vandermonde<T>(nodes, d);
    const double kappa = condition_bound(v);
    std::vector<double> c(static_cast<std::size_t>(d) + 1);
    for (std::size_t j = 0; j < c.size(); ++j) {
      c[j] = dyadic(j, 2);
    }
    std::vector<T> y;
    for (const auto& row : v) {
      Rational s = 0;
      for (std::size_t j = 0; j < c.size(); ++j) {
        s += row[j] * Rational(c[j]);
      }
      y.push_back(static_cast<T>(to_double(s)));
    }
    const NdArray<T> pinv = pb::bernstein_pinv<T>(std::span<const T>(nodes), d, std::nullopt);
    REQUIRE(pinv.shape()[0] == static_cast<std::size_t>(d) + 1);
    REQUIRE(pinv.shape()[1] == static_cast<std::size_t>(m));
    const double bound = fit_slack * static_cast<double>(m + d + 2) * u * kappa;
    for (std::size_t j = 0; j < c.size(); ++j) {
      double s = 0.0;
      for (std::size_t i = 0; i < y.size(); ++i) {
        s += static_cast<double>(pinv.data()[j * y.size() + i]) * static_cast<double>(y[i]);
      }
      INFO("degree " << d << ", coefficient " << j);
      CHECK(std::fabs(s - c[j]) <= bound);
    }
    // Perturbed data: add a fixed alternating perturbation outside the model space.
    std::vector<double> yp(y.size());
    for (std::size_t i = 0; i < y.size(); ++i) {
      yp[i] = static_cast<double>(y[i]) + ((i % 2 == 0) ? 0.125 : -0.0625);
    }
    std::vector<Rational> fit(c.size());
    for (std::size_t j = 0; j < c.size(); ++j) {
      Rational s = 0;
      for (std::size_t i = 0; i < yp.size(); ++i) {
        s += Rational(static_cast<double>(pinv.data()[j * yp.size() + i])) * Rational(yp[i]);
      }
      fit[j] = s;
    }
    // Residual r = V fit - yp, exactly; then V^T r.
    std::vector<Rational> resid(yp.size());
    for (std::size_t i = 0; i < yp.size(); ++i) {
      Rational s = 0;
      for (std::size_t j = 0; j < fit.size(); ++j) {
        s += v[i][j] * fit[j];
      }
      resid[i] = s - Rational(yp[i]);
    }
    double worst = 0.0;
    for (std::size_t j = 0; j < fit.size(); ++j) {
      Rational s = 0;
      for (std::size_t i = 0; i < yp.size(); ++i) {
        s += v[i][j] * resid[i];
      }
      worst = (std::max)(worst, std::fabs(to_double(s)));
    }
    // Backward error of a stable least-squares solve: ||V^T r|| <= K (m + d) u ||V|| (||V|| ||fit||
    // +
    // ||y||), with ||V||_inf <= 1 (rows of Bernstein values sum to 1) and ||fit||, ||y|| <= 2.
    // The explicit pseudo-inverse multiplies this by kappa (K absorbs the rest).
    INFO("degree " << d << ", worst |V^T r| " << worst);
    CHECK(worst <= fit_slack * static_cast<double>(m + d + 2) * u * kappa * 4.0);
  }
}

TEST_CASE("bernstein_pinv: degree 0 is the mean and degree 1 at the two ends is the identity") {
  // Closed forms. V = ones(m, 1) has the single singular value sqrt(m), pinv = ones^T / m. V for
  // degree 1 at nodes {0, 1} is the 2 x 2 identity.
  const std::vector<double> nodes5 = {0.1, 0.3, 0.5, 0.7, 0.9};
  const NdArray<double> mean =
      pb::bernstein_pinv<double>(std::span<const double>(nodes5), 0, std::nullopt);
  REQUIRE(mean.shape()[0] == 1);
  REQUIRE(mean.shape()[1] == 5);
  for (std::size_t i = 0; i < 5; ++i) {
    CHECK(std::fabs(mean.data()[i] - 0.2) <= 8.0 * std::numeric_limits<double>::epsilon());
  }
  const std::vector<double> ends = {0.0, 1.0};
  const NdArray<double> id =
      pb::bernstein_pinv<double>(std::span<const double>(ends), 1, std::nullopt);
  const double want[4] = {1, 0, 0, 1};
  for (std::size_t i = 0; i < 4; ++i) {
    CHECK(std::fabs(id.data()[i] - want[i]) <= 8.0 * std::numeric_limits<double>::epsilon());
  }
}

TEST_CASE_TEMPLATE("bernstein_pinv: singular values below tol * sigma_max are dropped", T, float,
                   double) {
  // V for degree 1 at the nodes {0, 0, 1} is [[1, 0], [1, 0], [0, 1]], with singular values
  // sqrt 2 and 1, so sigma_1 / sigma_0 = 0.7071...
  //   tol 0.70 keeps both: pinv = (V^T V)^-1 V^T = [[1/2, 1/2, 0], [0, 0, 1]];
  //   tol 0.72 drops sigma_1: pinv = v_0 u_0^T / sigma_0 = [[1/2, 1/2, 0], [0, 0, 0]];
  //   tol 1 keeps only the singular values equal to sigma_max (here one): the same rank-1 matrix;
  //   tol above 1 drops them all: the zero matrix (every coefficient fitted to 0).
  // Bug caught: a flipped comparison (keeping the small values), a cut taken relative to the
  // smallest value, or a tolerance ignored.
  const double tol_eps = 64.0 * unit_roundoff<T>();
  const std::vector<T> nodes = {T(0), T(0), T(1)};
  const auto fit_with = [&](double tol) {
    return pb::bernstein_pinv<T>(std::span<const T>(nodes), 1, tol);
  };
  const double both[6] = {0.5, 0.5, 0.0, 0.0, 0.0, 1.0};
  const double first[6] = {0.5, 0.5, 0.0, 0.0, 0.0, 0.0};
  const double none[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  const auto expect = [&](const NdArray<T>& got, const double(&want)[6], const char* what) {
    REQUIRE(got.shape()[0] == 2);
    REQUIRE(got.shape()[1] == 3);
    for (std::size_t i = 0; i < 6; ++i) {
      INFO(what << ", entry " << i);
      CHECK(std::fabs(static_cast<double>(got.data()[i]) - want[i]) <= tol_eps);
    }
  };
  expect(fit_with(0.70), both, "tol 0.70");
  expect(fit_with(0.72), first, "tol 0.72");
  expect(fit_with(1.0), first, "tol 1");
  expect(fit_with(1.01), none, "tol 1.01");
  expect(fit_with(1e-30), both, "a tiny tolerance keeps both");
  // Degree 0 from one value per node: the single singular value is sigma_max itself, kept at tol 1
  // and dropped above it.
  const std::vector<T> five = {T(0.1), T(0.3), T(0.5), T(0.7), T(0.9)};
  const NdArray<T> kept = pb::bernstein_pinv<T>(std::span<const T>(five), 0, 1.0);
  const NdArray<T> dropped = pb::bernstein_pinv<T>(std::span<const T>(five), 0, 1.5);
  for (std::size_t i = 0; i < 5; ++i) {
    CHECK(std::fabs(static_cast<double>(kept.data()[i]) - 0.2) <= tol_eps);
    CHECK(dropped.data()[i] == T(0));
  }
}

TEST_CASE_TEMPLATE("bernstein_pinv: the default tolerance truncates a near-duplicate node", T,
                   float, double) {
  // Nodes {1/2, 1/2 + delta}: det V = delta, sigma_1 / sigma_0 ~ delta. The default cutoff is
  // 100 eps; delta = 2 eps is below it (rank 1: entries of order 1) and an explicit 1e-3 eps keeps
  // the direction (entries of order 1/delta). Bug caught: a default that does not truncate, which
  // returns coefficients of order 1/eps from data with noise of order eps.
  const T eps = std::numeric_limits<T>::epsilon();
  const T delta = T(4) * eps;  // representable beside 1/2 (spacing eps / 2)
  const std::vector<T> nodes = {T(0.5), T(0.5) + delta};
  REQUIRE(nodes[1] > nodes[0]);
  const NdArray<T> truncated = pb::bernstein_pinv<T>(std::span<const T>(nodes), 1, std::nullopt);
  const NdArray<T> full =
      pb::bernstein_pinv<T>(std::span<const T>(nodes), 1, 1e-3 * static_cast<double>(eps));
  double largest_truncated = 0.0;
  double largest_full = 0.0;
  for (std::size_t i = 0; i < 4; ++i) {
    largest_truncated =
        (std::max)(largest_truncated, std::fabs(static_cast<double>(truncated.data()[i])));
    largest_full = (std::max)(largest_full, std::fabs(static_cast<double>(full.data()[i])));
  }
  CHECK(largest_truncated < 10.0);
  CHECK(largest_full > 1.0 / (16.0 * static_cast<double>(delta)));
}

TEST_CASE("bernstein_pinv: a single node gives the identity, other sizes are refused") {
  const std::vector<double> one = {0.37};
  const NdArray<double> p =
      pb::bernstein_pinv<double>(std::span<const double>(one), 0, std::nullopt);
  REQUIRE(p.shape()[0] == 1);
  REQUIRE(p.shape()[1] == 1);
  CHECK(p.data()[0] == 1.0);
  const std::vector<double> three = {0.0, 0.5, 1.0};
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::bernstein_pinv<double>(std::span<const double>(three),
                                                                    -1, std::nullopt)),
                       "bernstein_pinv needs 0 <= degree < len(nodes), got degree -1 for 3 nodes.",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::bernstein_pinv<double>(std::span<const double>(three),
                                                                    3, std::nullopt)),
                       "bernstein_pinv needs 0 <= degree < len(nodes), got degree 3 for 3 nodes.",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(pb::bernstein_pinv<double>(std::span<const double>(one), 1, std::nullopt)),
      "bernstein_pinv needs 0 <= degree < len(nodes), got degree 1 for 1 nodes.",
      pantr::value_error);
  CHECK_THROWS_AS(
      static_cast<void>(pb::bernstein_pinv<double>(std::span<const double>(), 0, std::nullopt)),
      pantr::value_error);
}

// ---- fit_tensor
// ------------------------------------------------------------------------------------

TEST_CASE_TEMPLATE("fit_tensor: a 2D tensor-grid interpolation returns the coefficients", T, float,
                   double) {
  // Bug caught: a pseudo-inverse applied along the wrong axis, a rank component mixed into the
  // grid, an output shape that is not (degree + 1, rank). The data are exact samples (rational
  // evaluation of the encoded coefficients) at the tensor product of two Chebyshev grids with
  // different sizes, rank 2.
  const double u = unit_roundoff<T>();
  const int n0 = 3;
  const int n1 = 4;
  const std::size_t rank = 2;
  const std::vector<T> x0 = nodes_of<T>(n0 + 1);
  const std::vector<T> x1 = nodes_of<T>(n1 + 1);
  const Matrix v0 = vandermonde<T>(x0, n0);
  const Matrix v1 = vandermonde<T>(x1, n1);
  std::vector<double> c((n0 + 1) * (n1 + 1) * rank);
  for (std::size_t i = 0; i < c.size(); ++i) {
    c[i] = dyadic(i, 3);
  }
  std::vector<T> values((n0 + 1) * (n1 + 1) * rank);
  for (std::size_t i = 0; i < x0.size(); ++i) {
    for (std::size_t j = 0; j < x1.size(); ++j) {
      for (std::size_t r = 0; r < rank; ++r) {
        Rational s = 0;
        for (std::size_t a = 0; a < v0[0].size(); ++a) {
          for (std::size_t b = 0; b < v1[0].size(); ++b) {
            s += v0[i][a] * v1[j][b] * Rational(c[(a * v1[0].size() + b) * rank + r]);
          }
        }
        values[(i * x1.size() + j) * rank + r] = static_cast<T>(to_double(s));
      }
    }
  }
  const std::vector<std::size_t> shape = {x0.size(), x1.size(), rank};
  const std::vector<std::span<const T>> nodes = {std::span<const T>(x0), std::span<const T>(x1)};
  const pb::Bezier<T> fit =
      pb::fit_tensor<T>(std::span<const T>(values), shape, nodes, std::nullopt, std::nullopt);
  REQUIRE(fit.dim() == 2);
  CHECK(fit.degree(0) == static_cast<std::size_t>(n0));
  CHECK(fit.degree(1) == static_cast<std::size_t>(n1));
  CHECK(fit.cp_size() == rank);
  CHECK_FALSE(fit.rational());
  const double bound =
      fit_slack * static_cast<double>(n0 + n1 + 4) * u * condition_bound(v0) * condition_bound(v1);
  REQUIRE(fit.control_points().size() == c.size());
  for (std::size_t i = 0; i < c.size(); ++i) {
    CAPTURE(i);
    CHECK(std::fabs(static_cast<double>(fit.control_points()[i]) - c[i]) <= bound);
  }
}

TEST_CASE("fit_tensor: a 3D grid with different sizes round trips") {
  const double u = unit_roundoff<double>();
  const std::vector<int> n = {2, 3, 2};
  const std::size_t rank = 1;
  std::vector<std::vector<double>> xs;
  std::vector<Matrix> vs;
  for (const int k : n) {
    xs.push_back(nodes_of<double>(k + 1));
    vs.push_back(vandermonde<double>(xs.back(), k));
  }
  std::vector<double> c(3 * 4 * 3 * rank);
  for (std::size_t i = 0; i < c.size(); ++i) {
    c[i] = dyadic(i, 5);
  }
  std::vector<double> values(c.size());
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 4; ++j) {
      for (std::size_t k = 0; k < 3; ++k) {
        Rational s = 0;
        for (std::size_t a = 0; a < 3; ++a) {
          for (std::size_t b = 0; b < 4; ++b) {
            for (std::size_t d = 0; d < 3; ++d) {
              s += vs[0][i][a] * vs[1][j][b] * vs[2][k][d] * Rational(c[(a * 4 + b) * 3 + d]);
            }
          }
        }
        values[(i * 4 + j) * 3 + k] = to_double(s);
      }
    }
  }
  const std::vector<std::size_t> shape = {3, 4, 3, rank};
  const std::vector<std::span<const double>> nodes = {std::span<const double>(xs[0]),
                                                      std::span<const double>(xs[1]),
                                                      std::span<const double>(xs[2])};
  const pb::Bezier<double> fit = pb::fit_tensor<double>(std::span<const double>(values), shape,
                                                        nodes, std::nullopt, std::nullopt);
  REQUIRE(fit.dim() == 3);
  CHECK(fit.degree(0) == 2);
  CHECK(fit.degree(1) == 3);
  CHECK(fit.degree(2) == 2);
  const double bound = fit_slack * 12.0 * u * condition_bound(vs[0]) * condition_bound(vs[1]) *
                       condition_bound(vs[2]);
  for (std::size_t i = 0; i < c.size(); ++i) {
    CAPTURE(i);
    CHECK(std::fabs(fit.control_points()[i] - c[i]) <= bound);
  }
}

TEST_CASE("fit_tensor: a direction with one node passes through at degree 0") {
  // The single node (0.5) of direction 0 gives a degree-0 direction, whose coefficient is the value
  // itself (analytic); direction 1 is interpolated. Bug caught: a pass-through that drops or
  // reorders the other direction's data, or a 1 x 1 pseudo-inverse applied with the wrong shape.
  const std::vector<double> x1 = nodes_of<double>(4);
  const Matrix v1 = vandermonde<double>(x1, 3);
  const std::vector<double> c = {0.5, -0.25, 0.75, 0.125};
  std::vector<double> values(4);
  for (std::size_t j = 0; j < 4; ++j) {
    Rational s = 0;
    for (std::size_t b = 0; b < 4; ++b) {
      s += v1[j][b] * Rational(c[b]);
    }
    values[j] = to_double(s);
  }
  const std::vector<double> x0 = {0.5};
  const std::vector<std::size_t> shape = {1, 4, 1};
  const std::vector<std::span<const double>> nodes = {std::span<const double>(x0),
                                                      std::span<const double>(x1)};
  const double bound = fit_slack * 8.0 * unit_roundoff<double>() * condition_bound(v1);
  const pb::Bezier<double> fit = pb::fit_tensor<double>(std::span<const double>(values), shape,
                                                        nodes, std::nullopt, std::nullopt);
  REQUIRE(fit.dim() == 2);
  CHECK(fit.degree(0) == 0);
  CHECK(fit.degree(1) == 3);
  for (std::size_t j = 0; j < 4; ++j) {
    CHECK(std::fabs(fit.control_points()[j] - c[j]) <= bound);
  }
  // With the degrees given explicitly, and with the single node first or last.
  const std::vector<long long> degrees = {0, 3};
  const pb::Bezier<double> given =
      pb::fit_tensor<double>(std::span<const double>(values), shape, nodes, degrees, std::nullopt);
  for (std::size_t j = 0; j < 4; ++j) {
    CHECK(std::fabs(given.control_points()[j] - c[j]) <= bound);
  }
  const std::vector<std::size_t> swapped_shape = {4, 1, 1};
  const std::vector<std::span<const double>> swapped_nodes = {std::span<const double>(x1),
                                                              std::span<const double>(x0)};
  const pb::Bezier<double> swapped = pb::fit_tensor<double>(
      std::span<const double>(values), swapped_shape, swapped_nodes, std::nullopt, std::nullopt);
  CHECK(swapped.degree(0) == 3);
  CHECK(swapped.degree(1) == 0);
  for (std::size_t j = 0; j < 4; ++j) {
    CHECK(std::fabs(swapped.control_points()[j] - c[j]) <= bound);
  }
  // A degree above 0 cannot be fitted to one node.
  const std::vector<long long> too_high = {1, 3};
  CHECK_THROWS_AS(static_cast<void>(pb::fit_tensor<double>(std::span<const double>(values), shape,
                                                           nodes, too_high, std::nullopt)),
                  pantr::value_error);
}

TEST_CASE("fit_tensor: a degree below the node count is a least-squares fit per direction") {
  // Exact data of degree 2 on 6 nodes: the fit with degrees {2} gives back the 3 coefficients and
  // the output order is degree + 1, not the node count.
  const std::vector<double> x = nodes_of<double>(6);
  const Matrix v = vandermonde<double>(x, 2);
  const std::vector<double> c = {0.25, -0.5, 0.875};
  std::vector<double> values(6);
  for (std::size_t j = 0; j < 6; ++j) {
    Rational s = 0;
    for (std::size_t b = 0; b < 3; ++b) {
      s += v[j][b] * Rational(c[b]);
    }
    values[j] = to_double(s);
  }
  const std::vector<std::size_t> shape = {6, 1};
  const std::vector<std::span<const double>> nodes = {std::span<const double>(x)};
  const std::vector<long long> degrees = {2};
  const pb::Bezier<double> fit =
      pb::fit_tensor<double>(std::span<const double>(values), shape, nodes, degrees, std::nullopt);
  REQUIRE(fit.degree(0) == 2);
  const double bound = fit_slack * 12.0 * unit_roundoff<double>() * condition_bound(v);
  for (std::size_t j = 0; j < 3; ++j) {
    CHECK(std::fabs(fit.control_points()[j] - c[j]) <= bound);
  }
}

TEST_CASE("fit_tensor: inconsistent arguments are refused with the documented messages") {
  const std::vector<double> values(12, 1.0);
  const std::vector<double> x3 = {0.0, 0.5, 1.0};
  const std::vector<double> x4 = {0.0, 0.3, 0.6, 1.0};
  const std::vector<std::span<const double>> both = {std::span<const double>(x3),
                                                     std::span<const double>(x4)};
  const std::vector<std::span<const double>> one = {std::span<const double>(x3)};
  const std::vector<std::size_t> grid = {3, 4, 1};
  CHECK_THROWS_WITH_AS(
      static_cast<void>(pb::fit_tensor<double>(std::span<const double>(values),
                                               std::vector<std::size_t>{12}, one, std::nullopt,
                                               std::nullopt)),
      "fit_tensor: values need (*n_pts, rank) extents, one node array and one degree per "
      "direction.",
      pantr::value_error);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(pb::fit_tensor<double>(std::span<const double>(values), grid, one,
                                               std::nullopt, std::nullopt)),
      "fit_tensor: values need (*n_pts, rank) extents, one node array and one degree per "
      "direction.",
      pantr::value_error);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(pb::fit_tensor<double>(std::span<const double>(values), grid, both,
                                               std::vector<long long>{2}, std::nullopt)),
      "fit_tensor: values need (*n_pts, rank) extents, one node array and one degree per "
      "direction.",
      pantr::value_error);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(pb::fit_tensor<double>(std::span<const double>(values.data(), 11), grid,
                                               both, std::nullopt, std::nullopt)),
      "fit_tensor: values do not match their shape.", pantr::value_error);
  const std::vector<std::span<const double>> wrong = {std::span<const double>(x4),
                                                      std::span<const double>(x4)};
  CHECK_THROWS_WITH_AS(
      static_cast<void>(pb::fit_tensor<double>(std::span<const double>(values), grid, wrong,
                                               std::nullopt, std::nullopt)),
      "fit_tensor: direction 0 has 4 nodes for 3 values.", pantr::value_error);
  // A negative degree and a degree at or above the node count both surface the pinv's message.
  CHECK_THROWS_WITH_AS(
      static_cast<void>(pb::fit_tensor<double>(std::span<const double>(values), grid, both,
                                               std::vector<long long>{-1, 3}, std::nullopt)),
      doctest::Contains("bernstein_pinv needs 0 <= degree < len(nodes)"), pantr::value_error);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(pb::fit_tensor<double>(std::span<const double>(values), grid, both,
                                               std::vector<long long>{2, 4}, std::nullopt)),
      doctest::Contains("bernstein_pinv needs 0 <= degree < len(nodes)"), pantr::value_error);
}

// ---- fit_scattered
// ---------------------------------------------------------------------------------

TEST_CASE_TEMPLATE("fit_scattered: scattered samples of a polynomial give its coefficients back", T,
                   float, double) {
  // Bug caught: a tensor-product Vandermonde with the directions in the wrong order (the first
  // direction slowest), the pseudo-inverse applied to the wrong operand, rank components mixed.
  // 2D, degrees (2, 3), 12 coefficients from 40 scattered points on a 1/64 grid (rank 2), exact
  // data. kappa_2 of the 40 x 12 matrix comes from the exact normal equations.
  const double u = unit_roundoff<T>();
  Lcg rng;
  const std::size_t n_pts = 40;
  const std::size_t rank = 2;
  std::vector<T> pts(n_pts * 2);
  for (T& p : pts) {
    p = static_cast<T>(static_cast<double>(rng.next() % 65) / 64.0);
  }
  // The 12-column tensor Vandermonde in exact arithmetic, first direction slowest.
  Matrix vt;
  for (std::size_t i = 0; i < n_pts; ++i) {
    const auto b0 = bernstein_values(2, Rational(static_cast<double>(pts[i * 2])));
    const auto b1 = bernstein_values(3, Rational(static_cast<double>(pts[i * 2 + 1])));
    std::vector<Rational> row;
    for (const Rational& a : b0) {
      for (const Rational& b : b1) {
        row.push_back(a * b);
      }
    }
    vt.push_back(row);
  }
  const double kappa = condition_bound(vt);
  REQUIRE(1.0 / kappa > 100.0 * static_cast<double>(std::numeric_limits<T>::epsilon()));
  std::vector<double> c(12 * rank);
  for (std::size_t i = 0; i < c.size(); ++i) {
    c[i] = dyadic(i, 7);
  }
  std::vector<T> values(n_pts * rank);
  for (std::size_t i = 0; i < n_pts; ++i) {
    for (std::size_t r = 0; r < rank; ++r) {
      Rational s = 0;
      for (std::size_t j = 0; j < 12; ++j) {
        s += vt[i][j] * Rational(c[j * rank + r]);
      }
      values[i * rank + r] = static_cast<T>(to_double(s));
    }
  }
  const std::vector<long long> degrees = {2, 3};
  const pb::Bezier<T> fit =
      pb::fit_scattered<T>(pantr::mdview<const T, 2>(values.data(), n_pts, rank),
                           pantr::mdview<const T, 2>(pts.data(), n_pts, 2), degrees, std::nullopt);
  REQUIRE(fit.dim() == 2);
  CHECK(fit.degree(0) == 2);
  CHECK(fit.degree(1) == 3);
  CHECK(fit.cp_size() == rank);
  CHECK_FALSE(fit.rational());
  const double bound = fit_slack * static_cast<double>(n_pts + 12 + 2) * u * kappa;
  REQUIRE(fit.control_points().size() == c.size());
  for (std::size_t i = 0; i < c.size(); ++i) {
    CAPTURE(i);
    CHECK(std::fabs(static_cast<double>(fit.control_points()[i]) - c[i]) <= bound);
  }
}

TEST_CASE("fit_scattered: as many points as coefficients interpolates, one fewer is refused") {
  // The boundary of "Underdetermined": 4 coefficients (degrees {3}) from exactly 4 points is a
  // square system and is solved; from 3 it is not.
  const std::vector<double> pts = {0.0, 0.25, 0.75, 1.0};
  const std::vector<double> values = {1.0, 2.0, -1.0, 0.5};
  const std::vector<long long> degrees = {3};
  const pb::Bezier<double> fit = pb::fit_scattered<double>(
      pantr::mdview<const double, 2>(values.data(), 4, 1),
      pantr::mdview<const double, 2>(pts.data(), 4, 1), degrees, std::nullopt);
  CHECK(fit.degree(0) == 3);
  // It reproduces the data: evaluate the fitted Bernstein sum at the points (exact basis).
  for (std::size_t i = 0; i < 4; ++i) {
    const auto b = bernstein_values(3, Rational(pts[i]));
    Rational s = 0;
    for (std::size_t j = 0; j < 4; ++j) {
      s += b[j] * Rational(fit.control_points()[j]);
    }
    CHECK(std::fabs(to_double(s) - values[i]) <= 256.0 * unit_roundoff<double>() * 8.0);
  }
  CHECK_THROWS_WITH_AS(
      static_cast<void>(pb::fit_scattered<double>(
          pantr::mdview<const double, 2>(values.data(), 3, 1),
          pantr::mdview<const double, 2>(pts.data(), 3, 1), degrees, std::nullopt)),
      "Underdetermined: 4 coefficients but only 3 sample points.", pantr::value_error);
  // A tensor count is checked as the product: degrees (3, 3) need 16 points.
  const std::vector<double> pts2(20, 0.5);
  const std::vector<double> values2(10, 1.0);
  const std::vector<long long> degrees2 = {3, 3};
  CHECK_THROWS_WITH_AS(
      static_cast<void>(pb::fit_scattered<double>(
          pantr::mdview<const double, 2>(values2.data(), 10, 1),
          pantr::mdview<const double, 2>(pts2.data(), 10, 2), degrees2, std::nullopt)),
      doctest::Contains("Underdetermined"), pantr::value_error);
}

TEST_CASE("fit_scattered: a negative degree and mismatched arguments are refused") {
  const std::vector<double> pts = {0.0, 0.25, 0.75, 1.0};
  const std::vector<double> values = {1.0, 2.0, -1.0, 0.5};
  const auto v = pantr::mdview<const double, 2>(values.data(), 4, 1);
  const auto p = pantr::mdview<const double, 2>(pts.data(), 4, 1);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(pb::fit_scattered<double>(v, p, std::vector<long long>{-1}, std::nullopt)),
      "degree must be non-negative", pantr::value_error);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(
          pb::fit_scattered<double>(v, p, std::vector<long long>{2, 1}, std::nullopt)),
      "fit_scattered: values (n_pts, rank), points (n_pts, dim) and dim degrees must agree.",
      pantr::value_error);
  const auto fewer = pantr::mdview<const double, 2>(values.data(), 3, 1);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(
          pb::fit_scattered<double>(fewer, p, std::vector<long long>{1}, std::nullopt)),
      "fit_scattered: values (n_pts, rank), points (n_pts, dim) and dim degrees must agree.",
      pantr::value_error);
}

TEST_CASE("bernstein_pinv: a non-finite node is refused, as LAPACK's SVD refused it") {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  for (const double bad : {nan, inf, -inf}) {
    const std::vector<double> nodes{0.0, bad, 0.5, 1.0};
    CHECK_THROWS_WITH_AS(static_cast<void>(pb::bernstein_pinv<double>(nodes, 3, std::nullopt)),
                         "SVD did not converge", pantr::value_error);
  }
}
