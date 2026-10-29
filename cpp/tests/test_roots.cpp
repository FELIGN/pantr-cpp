// The root-finding seam and its Builtin provider (design §4.1, design/root_seam.md): the wire
// values of the claim and completeness codes, a provider contract suite templated over
// `RootProvider`, the duplicate pair classes, the completeness values, validation, the Newton cap,
// the batch and the monotone solvers.
//
// Oracles. Encoded truth: polynomials built as products of known linear factors with rational
// roots, converted to the Bernstein basis in exact rational arithmetic (Boost cpp_rational, with
// the textbook map b_i = sum_{j <= i} a_j C(i, j) / C(n, j) written out below, never pantr's own
// conversion) and rounded once to the storage type. Analytic: hand-computed iterates of the Newton
// cap, the linear polynomial of the real depth-cap input. Metamorphic: scaling the coefficients by
// a power of two. Invariants: sorted, distinct, in [0, 1], claim at most `max_claim`, a gap list
// that covers every root the result lacks.
//
// Accuracy bound for a root r of multiplicity m, with p = (t - r)^m g, G = |g(r)|, B the largest
// stored coefficient magnitude, n the degree, u the unit roundoff of the storage type:
//   * rounding the exact coefficients to the storage type perturbs p by at most 4 u B (a relative
//     error of at most 4 u per coefficient, the two-step rational-to-double-to-float conversion
//     included, and the Bernstein basis is a partition of unity);
//   * de Casteljau in the storage type evaluates p to within 3 n u B (each step is a convex
//     combination of two values within B, with three roundings in double and one in float);
//   * the zero test of clipping accepts |f| <= max(4 (n + 1) eps_double B, value_tol)
//     (clipping.hpp, `zero_tolerance`).
// A returned root x therefore has |p(x)| <= band, the sum of the three, and the bracket or interval
// it came from adds param_tol to its position. On the ball |t - r| <= clearance / 10, |g| >= G / 3
// (at most ten other roots counted with multiplicity: 0.9^10 > 1/3), so
//   |x - r| <= param_tol + (3 band / G)^(1 / m).
// The helper requires that bound to stay below clearance / 10, so it cannot pass vacuously loose.
#include <doctest/doctest.h>

#include <pantr/bezier/bezier.hpp>
#include <pantr/core/errors.hpp>
#include <pantr/core/exact_rational.hpp>
#include <pantr/core/mdspan.hpp>
#include <pantr/core/ndarray.hpp>
#include <pantr/roots/roots.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace pr = pantr::roots;
namespace prd = pantr::roots::detail;
using pantr::exact::Integer;
using pantr::exact::Rational;

// The wire values: they cross into Python and between extension modules and never change. A change
// is a compile error here, and the Python suite reads the same numbers from the binding.
static_assert(std::is_same_v<std::underlying_type_t<pr::Claim>, std::int32_t>);
static_assert(std::is_same_v<std::underlying_type_t<pr::Completeness>, std::int32_t>);
static_assert(static_cast<std::int32_t>(pr::Claim::none) == 0);
static_assert(static_cast<std::int32_t>(pr::Claim::computed) == 1);
static_assert(static_cast<std::int32_t>(pr::Claim::certified) == 2);
static_assert(static_cast<std::int32_t>(pr::Completeness::complete) == 0);
static_assert(static_cast<std::int32_t>(pr::Completeness::gap_localized) == 1);
static_assert(static_cast<std::int32_t>(pr::Completeness::gap_unlocalized) == 2);

namespace {

constexpr double eps_double = std::numeric_limits<double>::epsilon();

template <class T>
constexpr double unit_roundoff() {
  return static_cast<double>(std::numeric_limits<T>::epsilon()) / 2.0;
}

Rational rat(long long num, long long den = 1) { return Rational(Integer(num), Integer(den)); }

Rational pow2(int k) {
  Integer p = 1;
  p <<= static_cast<unsigned>(k < 0 ? -k : k);
  return k >= 0 ? Rational(p) : Rational(Integer(1), p);
}

Integer binomial(int n, int k) {
  Integer r = 1;
  for (int i = 1; i <= k; ++i) {
    r = r * (n - k + i) / i;  // exact: r * (n - k + i) is C(n - k + i, i - 1) * i
  }
  return r;
}

double to_double(const Rational& q) { return q.convert_to<double>(); }

Rational abs_of(const Rational& q) { return q < 0 ? Rational(-q) : q; }

// A linear factor (t - root), `mult` times. A root outside [0, 1] pads the degree without adding
// a root to find.
struct Factor {
  Rational root;
  int mult;
};

// A root of the encoded polynomial inside [0, 1] and what limits the accuracy it can be found to.
struct ExpectedRoot {
  double x;          // the root
  int mult;          // its multiplicity m
  double leading;    // G = |p(t) / (t - x)^m| at t = x
  double clearance;  // distance to the nearest other root of p, inside [0, 1] or not
};

struct Truth {
  int degree = 0;
  std::vector<Rational> bernstein;             // exact, degree + 1 entries
  std::vector<ExpectedRoot> in_unit_interval;  // ascending
};

Truth make_truth(const std::vector<Factor>& factors, const Rational& lead = 1) {
  // Monomial coefficients of lead * prod (t - r)^m, by repeated multiplication with (t - r).
  std::vector<Rational> a{lead};
  for (const Factor& f : factors) {
    for (int k = 0; k < f.mult; ++k) {
      std::vector<Rational> next(a.size() + 1, Rational(0));
      for (std::size_t j = 0; j < a.size(); ++j) {
        next[j + 1] += a[j];
        next[j] -= f.root * a[j];
      }
      a = std::move(next);
    }
  }
  Truth tr;
  tr.degree = static_cast<int>(a.size()) - 1;
  tr.bernstein.assign(a.size(), Rational(0));
  for (int i = 0; i <= tr.degree; ++i) {
    for (int j = 0; j <= i; ++j) {
      tr.bernstein[static_cast<std::size_t>(i)] +=
          a[static_cast<std::size_t>(j)] * Rational(binomial(i, j), binomial(tr.degree, j));
    }
  }
  for (std::size_t i = 0; i < factors.size(); ++i) {
    if (factors[i].root < 0 || factors[i].root > 1) {
      continue;
    }
    Rational leading = abs_of(lead);
    Rational clearance = 1000;
    for (std::size_t j = 0; j < factors.size(); ++j) {
      if (j == i) {
        continue;
      }
      const Rational dist = abs_of(factors[i].root - factors[j].root);
      for (int k = 0; k < factors[j].mult; ++k) {
        leading *= dist;
      }
      clearance = std::min(clearance, dist);
    }
    tr.in_unit_interval.push_back(
        {to_double(factors[i].root), factors[i].mult, to_double(leading), to_double(clearance)});
  }
  std::sort(tr.in_unit_interval.begin(), tr.in_unit_interval.end(),
            [](const ExpectedRoot& l, const ExpectedRoot& r) { return l.x < r.x; });
  return tr;
}

// The same polynomial scaled by a power of two so that its largest coefficient is in [0.5, 1): rows
// of a batch share one value_tol, so they need one scale.
Truth unit_scale_truth(const std::vector<Factor>& factors) {
  double big = 0.0;
  for (const Rational& q : make_truth(factors).bernstein) {
    big = (std::max)(big, std::fabs(to_double(q)));
  }
  return make_truth(factors, pow2(-(std::ilogb(big) + 1)));
}

template <class T>
std::vector<T> stored(const Truth& tr) {
  std::vector<T> out;
  out.reserve(tr.bernstein.size());
  for (const Rational& q : tr.bernstein) {
    out.push_back(static_cast<T>(to_double(q)));
  }
  return out;
}

template <class T>
double largest_magnitude(const std::vector<T>& c) {
  double big = 0.0;
  for (const T v : c) {
    big = (std::max)(big, std::fabs(static_cast<double>(v)));
  }
  return big;
}

// The coefficient range the dispatch compares to its limit, and whether it then clips.
template <class T>
bool takes_clipping(const std::vector<T>& c) {
  double lo = std::numeric_limits<double>::infinity();
  for (const T v : c) {
    if (v != T(0)) {
      lo = (std::min)(lo, std::fabs(static_cast<double>(v)));
    }
  }
  return c.size() - 1 >= prd::clip_min_degree &&
         largest_magnitude(c) / lo <= prd::clip_coeff_range_limit;
}

struct Frame {
  double param_tol;
  double value_tol;
};

// value_tol: twice the evaluation-plus-rounding band, so that the zero test accepts a true root's
// own rounding noise in the storage type (a bare 1e-12 would reject every float root at the
// zero test, whose floor scales with eps_double only).
template <class T>
Frame frame_for(const Truth& tr, const std::vector<T>& coeff) {
  return {1e-10, 4.0 * (3.0 * tr.degree + 4.0) * unit_roundoff<T>() * largest_magnitude(coeff)};
}

template <class T>
double band_of(const Truth& tr, const std::vector<T>& coeff, Frame f) {
  const double big = largest_magnitude(coeff);
  const double n = tr.degree;
  const double u = unit_roundoff<T>();
  const double zero_floor = (std::max)(4.0 * (n + 1.0) * eps_double * big, f.value_tol);
  return zero_floor + 3.0 * n * u * big + 4.0 * u * big;
}

// The base merge radius of design/root_seam.md's pair-class table, max(2 param_tol, 4 zero_tol),
// with zero_tol = max(4 (n + 1) eps_double B, value_tol).
template <class T>
double merge_base_radius(const Truth& tr, const std::vector<T>& c, Frame f) {
  const double zero_tol =
      (std::max)(4.0 * (static_cast<double>(tr.degree) + 1.0) * eps_double * largest_magnitude(c),
                 f.value_tol);
  return (std::max)(2.0 * f.param_tol, 4.0 * zero_tol);
}

template <class T>
double position_bound(const Truth& tr, const std::vector<T>& coeff, const ExpectedRoot& e,
                      Frame f) {
  REQUIRE(tr.degree <= 11);  // the "at most ten other roots" of the bound
  const double band = band_of(tr, coeff, f);
  const double rho = f.param_tol + std::pow(3.0 * band / e.leading, 1.0 / e.mult);
  INFO("root " << e.x << " of multiplicity " << e.mult << ": bound " << rho << ", clearance "
               << e.clearance);
  REQUIRE(rho <= e.clearance / 10.0);
  return rho;
}

std::string listed(const std::vector<pr::Root<double>>& roots) {
  std::string s = "[";
  for (const auto& r : roots) {
    s += std::to_string(r.x) + " ";
  }
  return s + "]";
}

template <class T>
void check_roots_match(const pr::RootSet<double>& rs, const Truth& tr, const std::vector<T>& coeff,
                       Frame f) {
  INFO("found " << listed(rs.roots) << " for " << tr.in_unit_interval.size() << " roots");
  REQUIRE(rs.roots.size() == tr.in_unit_interval.size());
  for (std::size_t i = 0; i < rs.roots.size(); ++i) {
    const ExpectedRoot& e = tr.in_unit_interval[i];
    const double rho = position_bound<T>(tr, coeff, e, f);
    CAPTURE(i);
    CAPTURE(rs.roots[i].x - e.x);
    CHECK(std::fabs(rs.roots[i].x - e.x) <= rho);
  }
}

// What every RootSet of a provider with these capabilities satisfies, whatever the input.
void check_invariants(const pr::Capabilities& caps, const pr::RootSet<double>& rs, Frame f) {
  for (std::size_t i = 0; i < rs.roots.size(); ++i) {
    const pr::Root<double>& r = rs.roots[i];
    CAPTURE(i);
    CHECK(r.x >= 0.0);
    CHECK(r.x <= 1.0);
    CHECK(static_cast<int>(r.claim) >= static_cast<int>(pr::Claim::none));
    CHECK(static_cast<int>(r.claim) <= static_cast<int>(caps.max_claim));
    if (caps.multiplicity) {
      CHECK(r.multiplicity >= 1);
    } else {
      CHECK(r.multiplicity == -1);
    }
    if (i > 0) {
      CHECK(r.x - rs.roots[i - 1].x > f.param_tol);  // ascending and distinct within param_tol
    }
  }
  if (rs.completeness == pr::Completeness::complete) {
    CHECK(rs.gaps.empty());
  }
  if (rs.completeness == pr::Completeness::gap_localized) {
    CHECK_FALSE(rs.gaps.empty());
  }
  for (const auto& g : rs.gaps) {
    CHECK(g.first >= 0.0);
    CHECK(g.first < g.second);
    CHECK(g.second <= 1.0);
  }
}

template <class T>
pr::RootSet<double> builtin_roots(const std::vector<T>& c, Frame f) {
  return pr::Builtin{}.find_roots<T>(std::span<const T>(c), {f.param_tol, f.value_tol});
}

// The candidates clipping hands to the merge, and how many sit within `radius` of `x`.
template <class T>
prd::ClipResult raw_candidates(const std::vector<T>& c, Frame f) {
  return prd::clip_roots<T>(std::span<const T>(c), f.param_tol, f.value_tol);
}

int count_near(const std::vector<double>& xs, double x, double radius) {
  return static_cast<int>(
      std::count_if(xs.begin(), xs.end(), [&](double v) { return std::fabs(v - x) <= radius; }));
}

int count_near(const std::vector<pr::Root<double>>& rs, double x, double radius) {
  return static_cast<int>(std::count_if(rs.begin(), rs.end(), [&](const pr::Root<double>& v) {
    return std::fabs(v.x - x) <= radius;
  }));
}

// A named input of the catalogue: roots k/16 and friends, sized to take a known rung.
struct Spec {
  const char* name;
  std::vector<Factor> factors;
  bool clipping;  // the rung the input is built to take
};

std::vector<Spec> catalogue() {
  const Factor out1{rat(-1), 1};
  const Factor out2{rat(2), 1};
  const Factor out3{rat(3), 1};
  const Factor outm2{rat(-2), 1};
  return {
      {"linear, root 1/3", {{rat(1, 3), 1}}, false},
      {"quadratic, roots 1/4 and 3/4", {{rat(1, 4), 1}, {rat(3, 4), 1}}, false},
      {"cubic, roots 0, 1/2 and 1", {{rat(0), 1}, {rat(1, 2), 1}, {rat(1), 1}}, false},
      {"quintic, roots 1/8 3/8 5/8 7/8 and one outside",
       {{rat(1, 8), 1}, {rat(3, 8), 1}, {rat(5, 8), 1}, {rat(7, 8), 1}, out2},
       false},
      {"quintic without a root in [0, 1]", {out1, out2, out3, outm2, {rat(4), 1}}, false},
      {"degree 7, roots k/16 (clipping)",
       {{rat(2, 16), 1},
        {rat(3, 16), 1},
        {rat(5, 16), 1},
        {rat(8, 16), 1},
        {rat(11, 16), 1},
        {rat(13, 16), 1},
        {rat(14, 16), 1}},
       true},
      {"degree 8, five roots and three outside (clipping)",
       {{rat(1, 16), 1},
        {rat(4, 16), 1},
        {rat(8, 16), 1},
        {rat(12, 16), 1},
        {rat(15, 16), 1},
        out1,
        out2,
        out3},
       true},
      {"degree 6, roots at 0 and 1 and 1/2 (clipping)",
       {{rat(0), 1}, {rat(1), 1}, {rat(1, 2), 1}, out1, out2, out3},
       true},
      {"degree 6 without a root in [0, 1] (clipping)",
       {out1, out2, out3, outm2, {rat(4), 1}, {rat(-3), 1}},
       true},
  };
}

// A provider that proves the seam is satisfiable by another one, and records what the entry point
// handed it.
struct RecordingProvider {
  mutable std::vector<double> seen;
  mutable pr::FindRootsOptions options{0.0, 0.0};

  [[nodiscard]] pr::Capabilities capabilities() const {
    return {pr::Claim::certified, false, false, true};
  }

  template <class T>
  [[nodiscard]] pr::RootSet<double> find_roots(std::span<const T> c, pr::FindRootsOptions o) const {
    seen.assign(c.begin(), c.end());
    options = o;
    return {{{0.25, pr::Claim::certified, 1}}, pr::Completeness::complete, {}};
  }
};

struct NotAProvider {
  template <class T>
  [[nodiscard]] pr::RootSet<double> find_roots(std::span<const T>, pr::FindRootsOptions) const {
    return {{}, pr::Completeness::complete, {}};
  }
};

template <class P, class S>
struct ProviderCase {
  using Provider = P;
  using Scalar = S;
};

// Add a provider here and the whole contract suite runs for it.
using ProviderCases =
    std::tuple<ProviderCase<pr::Builtin, float>, ProviderCase<pr::Builtin, double>>;

}  // namespace

DOCTEST_TYPE_TO_STRING_AS("Builtin<float>", ProviderCase<pr::Builtin, float>);
DOCTEST_TYPE_TO_STRING_AS("Builtin<double>", ProviderCase<pr::Builtin, double>);

static_assert(pr::RootProvider<RecordingProvider, double>);
static_assert(!pr::RootProvider<NotAProvider, double>);  // no capabilities(): not a provider

TEST_CASE("roots: the claim and completeness codes keep their wire values") {
  CHECK(static_cast<int>(pr::Claim::none) == 0);
  CHECK(static_cast<int>(pr::Claim::computed) == 1);
  CHECK(static_cast<int>(pr::Claim::certified) == 2);
  CHECK(static_cast<int>(pr::Completeness::complete) == 0);
  CHECK(static_cast<int>(pr::Completeness::gap_localized) == 1);
  CHECK(static_cast<int>(pr::Completeness::gap_unlocalized) == 2);
}

TEST_CASE("roots: Builtin states what it can do") {
  const pr::Capabilities caps = pr::Builtin{}.capabilities();
  CHECK(caps.max_claim == pr::Claim::computed);
  CHECK(caps.batch);
  CHECK(caps.monotone);
  CHECK_FALSE(caps.multiplicity);
}

TEST_CASE("roots: the encoded truth is exact (the oracle itself)") {
  // (t - 1/4)(t - 3/4) = t^2 - t + 3/16 has Bernstein coefficients (3/16, -5/16, 3/16): b_1 =
  // a_0 + a_1 / 2 = 3/16 - 1/2, and p(1/4) = 0 by de Casteljau on the exact values.
  const Truth tr = make_truth({{rat(1, 4), 1}, {rat(3, 4), 1}});
  REQUIRE(tr.bernstein.size() == 3);
  CHECK(tr.bernstein[0] == rat(3, 16));
  CHECK(tr.bernstein[1] == rat(-5, 16));
  CHECK(tr.bernstein[2] == rat(3, 16));
  // A degree-5 input with a double root: the exact Bernstein form reproduces the factored value at
  // an interior point through the Bernstein sum.
  const Truth q = make_truth({{rat(1, 2), 2}, {rat(1, 8), 1}, {rat(-1), 1}, {rat(2), 1}});
  const Rational t = rat(3, 10);
  Rational sum = 0;
  Rational factored = 1;
  for (int i = 0; i <= q.degree; ++i) {
    Rational basis = Rational(binomial(q.degree, i));
    for (int k = 0; k < i; ++k) {
      basis *= t;
    }
    for (int k = 0; k < q.degree - i; ++k) {
      basis *= (1 - t);
    }
    sum += q.bernstein[static_cast<std::size_t>(i)] * basis;
  }
  factored = (t - rat(1, 2)) * (t - rat(1, 2)) * (t - rat(1, 8)) * (t + 1) * (t - 2);
  CHECK(sum == factored);
}

TEST_CASE_TEMPLATE_DEFINE(
    "roots contract: encoded-truth polynomials are recovered, sorted, in range", Case,
    roots_contract) {
  // Bug caught: a root lost, doubled or misplaced; an unsorted or out-of-range result; a claim
  // above the provider's own `max_claim`; a gap list that disagrees with the completeness code; a
  // result that changes with the scale of the coefficients (each input is run at 2^-10, 1 and 2^10;
  // the merge radius that limits the scale is the subject of the frontier test below).
  using P = typename Case::Provider;
  using T = typename Case::Scalar;
  static_assert(pr::RootProvider<P, T>);
  const P provider{};
  const pr::Capabilities caps = provider.capabilities();
  for (const Spec& spec : catalogue()) {
    for (const int shift : {-10, 0, 10}) {
      const Truth tr = make_truth(spec.factors, pow2(shift));
      const std::vector<T> c = stored<T>(tr);
      const Frame f = frame_for<T>(tr, c);
      INFO(spec.name << ", scaled by 2^" << shift);
      REQUIRE(takes_clipping(c) == spec.clipping);  // which rung answers is part of the case
      const pr::RootSet<double> rs = provider.find_roots(
          std::span<const T>(c), pr::FindRootsOptions{f.param_tol, f.value_tol});
      check_invariants(caps, rs, f);
      CHECK(rs.completeness == pr::Completeness::complete);
      check_roots_match<T>(rs, tr, c, f);
    }
  }
}

TEST_CASE_TEMPLATE_APPLY(roots_contract, ProviderCases);

TEST_CASE_TEMPLATE("roots contract: reflecting the parameter reflects the roots", T, float,
                   double) {
  // Metamorphic: p(1 - t) has the reversed coefficients and the roots 1 - r. Bug caught: a finder
  // that is accurate at one end of [0, 1] and not the other (an index shifted by one in a hull
  // clip or a bracket update).
  for (const Spec& spec : catalogue()) {
    const Truth tr = make_truth(spec.factors);
    std::vector<T> c = stored<T>(tr);
    std::reverse(c.begin(), c.end());
    std::vector<Factor> mirrored = spec.factors;
    for (Factor& fac : mirrored) {
      fac.root = 1 - fac.root;
    }
    const Truth mirror = make_truth(mirrored, tr.degree % 2 == 0 ? rat(1) : rat(-1));
    // Reversing the coefficients maps p(t) to p(1 - t) = lead * prod (1 - t - r) = (-1)^n prod (t -
    // (1 - r)), so the mirrored truth carries the sign (-1)^n.
    const std::vector<T> exact_mirror = stored<T>(mirror);
    REQUIRE(exact_mirror.size() == c.size());
    for (std::size_t i = 0; i < c.size(); ++i) {
      CHECK(exact_mirror[i] == c[i]);  // the same rounding of the same exact value
    }
    const Frame f = frame_for<T>(mirror, c);
    INFO(spec.name);
    check_roots_match<T>(builtin_roots<T>(c, f), mirror, c, f);
  }
}

// ---- The duplicate pair classes (design/root_seam.md): one root per true root ------------------

TEST_CASE_TEMPLATE("roots dedup (a): a root at a subdivision point shared by two intervals", T,
                   float, double) {
  // Bug caught: the merge keeping both copies of a root that clipping reports from each of the two
  // halves it meets at 1/2 (a flipped or too-small comparison against the merge radius).
  const Truth tr = make_truth({{rat(2, 16), 1},
                               {rat(3, 16), 1},
                               {rat(5, 16), 1},
                               {rat(8, 16), 1},
                               {rat(11, 16), 1},
                               {rat(13, 16), 1},
                               {rat(14, 16), 1}});
  const std::vector<T> c = stored<T>(tr);
  const Frame f = frame_for<T>(tr, c);
  REQUIRE(takes_clipping(c));
  const double rho = position_bound<T>(tr, c, tr.in_unit_interval[3], f);
  const prd::ClipResult raw = raw_candidates<T>(c, f);
  // The pair is produced: without it this test would pass over a polynomial the merge never sees.
  REQUIRE(count_near(raw.candidates, 0.5, rho) >= 2);
  const pr::RootSet<double> rs = builtin_roots<T>(c, f);
  CHECK(count_near(rs.roots, 0.5, rho) == 1);
  check_roots_match<T>(rs, tr, c, f);
}

TEST_CASE("roots dedup (b): a double root reached by several clip branches is reported once") {
  // Bug caught: the derivative-aware radius dropped or capped too low, so the candidates a double
  // root spreads over (here the band |f| <= zero_tol) come out as several roots.
  const Truth tr =
      make_truth({{rat(2, 5), 2}, {rat(-1), 1}, {rat(2), 1}, {rat(3), 1}, {rat(-2), 1}});
  const std::vector<double> c = stored<double>(tr);
  const Frame f{1e-10, 1e-12};
  REQUIRE(takes_clipping(c));
  const ExpectedRoot& e = tr.in_unit_interval[0];
  const double rho = position_bound<double>(tr, c, e, f);
  const prd::ClipResult raw = raw_candidates<double>(c, f);
  // Reached: at least two candidates farther apart than the base merge radius, so that only the
  // derivative-aware radius can join them (the base radius is max(2 param_tol, 4 zero_tol)).
  const double base = merge_base_radius(tr, c, f);
  std::vector<double> clustered;
  for (const double x : raw.candidates) {
    if (std::fabs(x - e.x) <= rho) {
      clustered.push_back(x);
    }
  }
  REQUIRE(clustered.size() >= 2);
  REQUIRE(*std::max_element(clustered.begin(), clustered.end()) -
              *std::min_element(clustered.begin(), clustered.end()) >
          base);
  const pr::RootSet<double> rs = builtin_roots<double>(c, f);
  CHECK(count_near(rs.roots, e.x, rho) == 1);
  check_roots_match<double>(rs, tr, c, f);
}

TEST_CASE("roots dedup (b): a triple root reached by several clip branches is reported once") {
  // As above at multiplicity 3, where the radius cap zero_tol^(1/3) (not the derivative) must
  // carry the merge.
  const Truth tr =
      make_truth({{rat(2, 5), 3}, {rat(-1), 1}, {rat(2), 1}, {rat(3), 1}, {rat(-2), 1}});
  const std::vector<double> c = stored<double>(tr);
  const Frame f{1e-10, 1e-12};
  REQUIRE(takes_clipping(c));
  const ExpectedRoot& e = tr.in_unit_interval[0];
  const double rho = position_bound<double>(tr, c, e, f);
  const prd::ClipResult raw = raw_candidates<double>(c, f);
  const double base = merge_base_radius(tr, c, f);
  std::vector<double> clustered;
  for (const double x : raw.candidates) {
    if (std::fabs(x - e.x) <= rho) {
      clustered.push_back(x);
    }
  }
  REQUIRE(clustered.size() >= 2);
  REQUIRE(*std::max_element(clustered.begin(), clustered.end()) -
              *std::min_element(clustered.begin(), clustered.end()) >
          base);
  const pr::RootSet<double> rs = builtin_roots<double>(c, f);
  CHECK(count_near(rs.roots, e.x, rho) == 1);
  check_roots_match<double>(rs, tr, c, f);
}

TEST_CASE("roots dedup (b): a triple root at 1/2 with distant simple roots is reported once") {
  // The subdivision point and the multiple root together: every candidate lands on 0.5 exactly.
  const Truth tr = make_truth(
      {{rat(1, 2), 3}, {rat(1, 5), 1}, {rat(4, 5), 1}, {rat(1, 10), 1}, {rat(9, 10), 1}});
  const std::vector<double> c = stored<double>(tr);
  const Frame f{1e-10, 1e-12};
  REQUIRE(takes_clipping(c));
  const ExpectedRoot& e = tr.in_unit_interval[2];
  const double rho = position_bound<double>(tr, c, e, f);
  REQUIRE(count_near(raw_candidates<double>(c, f).candidates, e.x, rho) >= 2);
  const pr::RootSet<double> rs = builtin_roots<double>(c, f);
  CHECK(count_near(rs.roots, e.x, rho) == 1);
  check_roots_match<double>(rs, tr, c, f);
}

TEST_CASE("roots dedup (b) frontier: a triple root with a small leading factor is reported once" *
          doctest::should_fail()) {
  // KNOWN DEFECT, inherited bit for bit from the previous (Numba) release, which reports the
  // triple root twice for this input: the merge radius at a triple root,
  // min(4 zero_tol / |f'|, zero_tol^(1/3)), is smaller than the spread of the candidates when
  // G = |g(r)| is small, because the first kept candidate sits near the edge of the band |f| <=
  // zero_tol and the next on the other side. design/root_seam.md claims the class covered for m <=
  // 3. Remove `should_fail` when the merge is repaired; this test then pins the repair.
  const Truth tr = make_truth({{rat(2, 5), 3}, {rat(1, 10), 1}, {rat(9, 10), 1}, {rat(-1), 1}});
  const std::vector<double> c = stored<double>(tr);
  const Frame f{1e-10, 1e-12};
  REQUIRE(takes_clipping(c));
  const ExpectedRoot& e = tr.in_unit_interval[1];
  const double rho = position_bound<double>(tr, c, e, f);
  const pr::RootSet<double> rs = builtin_roots<double>(c, f);
  CHECK(count_near(rs.roots, e.x, rho) == 1);
}

TEST_CASE_TEMPLATE("roots dedup (c): a root at 0 or 1 found twice is reported once", T, float,
                   double) {
  // Bug caught: the boundary check before clipping and the polished midpoint of the last interval
  // both reporting the endpoint (a merge radius of zero, or a polish that leaves [0, 1]).
  struct Case {
    const char* name;
    std::vector<Factor> factors;
  };
  const std::vector<Case> cases = {
      {"root at 0",
       {{rat(0), 1},
        {rat(1, 4), 1},
        {rat(3, 4), 1},
        {rat(-1), 1},
        {rat(2), 1},
        {rat(3), 1},
        {rat(-2), 1}}},
      {"root at 1",
       {{rat(1), 1},
        {rat(1, 4), 1},
        {rat(3, 4), 1},
        {rat(-1), 1},
        {rat(2), 1},
        {rat(3), 1},
        {rat(-2), 1}}},
      {"roots at 0 and 1",
       {{rat(0), 1},
        {rat(1), 1},
        {rat(1, 2), 1},
        {rat(-1), 1},
        {rat(2), 1},
        {rat(3), 1},
        {rat(-2), 1}}},
  };
  for (const Case& cs : cases) {
    INFO(cs.name);
    const Truth tr = make_truth(cs.factors);
    const std::vector<T> c = stored<T>(tr);
    const Frame f = frame_for<T>(tr, c);
    REQUIRE(takes_clipping(c));
    const prd::ClipResult raw = raw_candidates<T>(c, f);
    for (const ExpectedRoot& e : tr.in_unit_interval) {
      if (e.x == 0.0 || e.x == 1.0) {
        const double rho = position_bound<T>(tr, c, e, f);
        REQUIRE(count_near(raw.candidates, e.x, rho) >= 2);  // the pair is produced
        CHECK(count_near(builtin_roots<T>(c, f).roots, e.x, rho) == 1);
      }
    }
    check_roots_match<T>(builtin_roots<T>(c, f), tr, c, f);
  }
}

TEST_CASE("roots dedup: two distinct simple roots 1e-5 apart are not merged") {
  // The other half of the pair-class table: merging duplicates must not merge distinct roots. The
  // pair 2/5 and 2/5 + 1e-5 is 5 orders of magnitude above the base merge radius and below the cap
  // zero_tol^(1/3), so only the derivative-aware radius separates it from a duplicate: at a simple
  // root |f'| is about 1e-5 times the other factors, which makes 4 zero_tol / |f'| tiny. Bug
  // caught: a radius that forgets the derivative and so merges everything under the cap (about 3e-5
  // here). Two rungs: degree 6 clips, degree 4 goes to Yuksel (single calls do not merge there, a
  // batch row does, and must keep both).
  const Frame f{1e-10, 1e-15};
  const Rational near_pair = rat(40001, 100000);
  const Truth clip = make_truth(
      {{rat(2, 5), 1}, {near_pair, 1}, {rat(-1), 1}, {rat(2), 1}, {rat(3), 1}, {rat(-2), 1}});
  const std::vector<double> cc = stored<double>(clip);
  REQUIRE(takes_clipping(cc));
  check_roots_match<double>(builtin_roots<double>(cc, f), clip, cc, f);
  const Truth yuk = make_truth({{rat(2, 5), 1}, {near_pair, 1}, {rat(-1), 1}, {rat(2), 1}});
  const std::vector<double> yc = stored<double>(yuk);
  REQUIRE_FALSE(takes_clipping(yc));
  check_roots_match<double>(builtin_roots<double>(yc, f), yuk, yc, f);
  pantr::NdArray<double> m({1, yc.size()});
  std::copy(yc.begin(), yc.end(), m.data());
  const pr::BatchRootSet br = pr::Builtin{}.find_roots_batch<double>(
      pantr::mdview<const double, 2>(m.data(), 1, yc.size()), {f.param_tol, f.value_tol});
  CHECK(br.counts[0] == 2);
}

TEST_CASE("roots dedup (c)+(b): a double root at 0 and a triple root at 1 are reported once each") {
  const Truth tr = make_truth({{rat(0), 2}, {rat(1), 3}, {rat(1, 2), 1}, {rat(-1), 1}});
  const std::vector<double> c = stored<double>(tr);
  const Frame f{1e-10, 1e-12};
  REQUIRE(takes_clipping(c));
  const pr::RootSet<double> rs = builtin_roots<double>(c, f);
  CHECK(rs.roots.size() == 3);
  check_roots_match<double>(rs, tr, c, f);
}

TEST_CASE("roots dedup (d): a double root at a critical point of Yuksel's walk is reported once") {
  // Bug caught: the walk's guard against an endpoint within tol of the last root removed, so that
  // the root at a critical point (where two adjacent monotone intervals meet) is pushed twice.
  struct Case {
    const char* name;
    std::vector<Factor> factors;
  };
  const std::vector<Case> cases = {
      {"double root at 1/2 of a quadratic", {{rat(1, 2), 2}}},
      {"double root at 1/2 between 1/8 and 7/8", {{rat(1, 2), 2}, {rat(1, 8), 1}, {rat(7, 8), 1}}},
      {"double root at 1/2, degree 5",
       {{rat(1, 2), 2}, {rat(1, 8), 1}, {rat(7, 8), 1}, {rat(2), 1}}},
      {"double root at 0 and a simple root", {{rat(0), 2}, {rat(1, 2), 1}}},
      {"double root at 1 and a simple root", {{rat(1), 2}, {rat(1, 2), 1}}},
      // The walk's guard is load-bearing here (searched over double and triple roots at k/64 with
      // up to three other roots, it is one of very few such placements): without it the triple root
      // is pushed once by the solve on the interval it ends and once by the endpoint test.
      {"triple root at 37/64 with roots at 7/8 and -2",
       {{rat(37, 64), 3}, {rat(7, 8), 1}, {rat(-2), 1}}},
  };
  for (const Case& cs : cases) {
    INFO(cs.name);
    const Truth tr = make_truth(cs.factors);
    const std::vector<double> c = stored<double>(tr);
    REQUIRE_FALSE(takes_clipping(c));
    const Frame f{1e-10, 1e-12};
    check_roots_match<double>(builtin_roots<double>(c, f), tr, c, f);
  }
}

TEST_CASE(
    "roots dedup (d) frontier: a double root on the single-call Yuksel path is reported once" *
    doctest::should_fail()) {
  // KNOWN DEFECT, inherited from the previous release: the same defect as the triple root below,
  // at its most ordinary. (t - 3/16)^2 (t + 1) (t - 2) has one root in [0, 1], and the single call
  // reports it twice, by the sign of the rounding noise of f at the critical point; the guard of
  // the walk (dedup (d) above) removes some placements and not others. Remove `should_fail` when
  // the single call merges as the batch does.
  const Truth tr = make_truth({{rat(3, 16), 2}, {rat(-1), 1}, {rat(2), 1}});
  const std::vector<double> c = stored<double>(tr);
  REQUIRE_FALSE(takes_clipping(c));
  const Frame f{1e-10, 1e-12};
  const pr::RootSet<double> rs = builtin_roots<double>(c, f);
  CHECK(rs.roots.size() == 1);
}

TEST_CASE(
    "roots dedup (d) frontier: a triple root on the single-call Yuksel path is reported once" *
    doctest::should_fail()) {
  // KNOWN DEFECT, inherited from the previous release, which repeats the root for this input:
  // the single call does not merge Yuksel's roots, so a triple root (and two double roots, for
  // some placements) come out repeated, against design §4.1's "distinct within the request's
  // param_tol". The batch dispatch merges them (a later test). Remove `should_fail` when the
  // single call merges too.
  const Truth tr = make_truth({{rat(1, 2), 3}, {rat(1, 8), 1}, {rat(7, 8), 1}});
  const std::vector<double> c = stored<double>(tr);
  REQUIRE_FALSE(takes_clipping(c));
  const Frame f{1e-10, 1e-12};
  const pr::RootSet<double> rs = builtin_roots<double>(c, f);
  CHECK(rs.roots.size() == 3);
}

TEST_CASE("roots dedup (d): the batch dispatch merges Yuksel's roots that the single call keeps") {
  // The two dispatches differ by design ("kept from the previous release"): the same triple root
  // is one root in a batch row. Bug caught: the batch dropping its merge.
  const Truth tr = make_truth({{rat(1, 2), 3}, {rat(1, 8), 1}, {rat(7, 8), 1}});
  const std::vector<double> c = stored<double>(tr);
  pantr::NdArray<double> m({1, c.size()});
  std::copy(c.begin(), c.end(), m.data());
  const pr::BatchRootSet br = pr::Builtin{}.find_roots_batch<double>(
      pantr::mdview<const double, 2>(m.data(), 1, c.size()), {1e-10, 1e-12});
  CHECK(br.counts[0] == 3);
}

// ---- Scale
// ---------------------------------------------------------------------------------------

TEST_CASE("roots scale: distinct roots survive coefficients of order 1e9") {
  // Metamorphic (ST-rotate-shift-and-scale): scaling a polynomial leaves its roots alone.
  // 2^30 is the largest scale the merge radius max(2 param_tol, 4 zero_tol) still handles with
  // roots 1/16 apart; see the frontier test below for where it stops.
  const Truth tr = make_truth({{rat(2, 16), 1},
                               {rat(3, 16), 1},
                               {rat(5, 16), 1},
                               {rat(8, 16), 1},
                               {rat(11, 16), 1},
                               {rat(13, 16), 1},
                               {rat(14, 16), 1}},
                              pow2(30));
  const std::vector<double> c = stored<double>(tr);
  const Frame f = frame_for<double>(tr, c);
  check_roots_match<double>(builtin_roots<double>(c, f), tr, c, f);
}

TEST_CASE("roots scale frontier: distinct roots are not merged at coefficient scale 2^50" *
          doctest::should_fail()) {
  // KNOWN DEFECT, inherited from the previous release, which also loses most of these roots at
  // this scale: the merge radius `max(2 param_tol, 4 zero_tol)` compares a gap in the parameter
  // with `zero_tol`, a quantity in coefficient units (scale * (n + 1) * 4 * eps), so for
  // coefficients near 1e12 and above, roots 1/16 apart merge, silently and with completeness
  // `complete`. Remove `should_fail` when the radius is made scale-free (for example by dividing
  // the zero tolerance by the derivative, as the second radius does).
  const Truth tr = make_truth({{rat(2, 16), 1},
                               {rat(3, 16), 1},
                               {rat(5, 16), 1},
                               {rat(8, 16), 1},
                               {rat(11, 16), 1},
                               {rat(13, 16), 1},
                               {rat(14, 16), 1}},
                              pow2(50));
  const std::vector<double> c = stored<double>(tr);
  const Frame f = frame_for<double>(tr, c);
  const pr::RootSet<double> rs = builtin_roots<double>(c, f);
  CHECK(rs.roots.size() == 7);
}

// ---- Completeness ------------------------------------------------------------------------------

TEST_CASE("roots completeness: the public depth-cap input reports gap_localized with its gap") {
  // `Completeness::gap_unlocalized` has no test here, deliberately: it is for providers that know
  // a part of [0, 1] carries no claim and not where, and Builtin always knows the interval it gave
  // up on, so the value is unreachable through it and is not faked. The depth cap below is the
  // only gap Builtin produces, and it is localized.
  //
  // A linear polynomial elevated to degree 8, root 0.13881295865011115 (the analytic root of
  // t - 0.13881295865011115), with a tolerance below the spacing of doubles at the root, so no
  // interval can narrow to it and clipping reaches its depth cap. Bug caught: the cap passing
  // unremarked (the previous release's silent drop): the root is found and the result says
  // `complete`.
  const std::vector<double> c = {-0.13881295865011115, -0.013812958650111146, 0.11118704134988885,
                                 0.23618704134988885,  0.3611870413498889,    0.4861870413498889,
                                 0.6111870413498889,   0.7361870413498889,    0.8611870413498889};
  const double tol = 1.2388055965619009e-52;
  const double root = 0.13881295865011115;
  const pr::RootSet<double> rs = builtin_roots<double>(c, {tol, tol});
  CHECK(rs.completeness == pr::Completeness::gap_localized);
  REQUIRE(rs.gaps.size() == 1);
  // The coefficients are rounded to 1 ulp (at most 1.1e-16 * 0.87), the polynomial has slope 1, so
  // the root and the gap are within 4 eps_double of the analytic root.
  const double slack = 4.0 * eps_double;
  CHECK(rs.gaps[0].first <= root + slack);
  CHECK(rs.gaps[0].second >= root - slack);
  CHECK(rs.gaps[0].second - rs.gaps[0].first <= slack);
  REQUIRE(rs.roots.size() == 1);  // the interval is polished and kept if its residual allows
  CHECK(std::fabs(rs.roots[0].x - root) <= slack);
  check_invariants(pr::Builtin{}.capabilities(), rs, {tol, tol});
}

TEST_CASE("roots completeness: the depth cap reports every root it gives up on as a gap") {
  // Bug caught: an interval abandoned at `max_depth` that is neither a candidate nor a gap, so a
  // root silently disappears. Property: for every cap, each true root is a candidate or inside a
  // listed gap. Small caps reach the path on an ordinary polynomial (the default cap of 64 does
  // not).
  const Truth tr = make_truth({{rat(2, 16), 1},
                               {rat(3, 16), 1},
                               {rat(5, 16), 1},
                               {rat(8, 16), 1},
                               {rat(11, 16), 1},
                               {rat(13, 16), 1},
                               {rat(14, 16), 1}});
  const std::vector<double> c = stored<double>(tr);
  const Frame f = frame_for<double>(tr, c);
  for (const int depth : {0, 1, 2, 3, 4, 5}) {
    INFO("max_depth " << depth);
    const prd::ClipResult res =
        prd::clip_roots<double>(std::span<const double>(c), f.param_tol, f.value_tol, depth);
    if (depth <= 3) {
      CHECK_FALSE(res.gaps.empty());  // the cap is reached: the test is not vacuous
    }
    for (const auto& g : res.gaps) {
      CHECK(g.first >= 0.0);
      CHECK(g.first < g.second);
      CHECK(g.second <= 1.0);
    }
    for (const ExpectedRoot& e : tr.in_unit_interval) {
      const double rho = position_bound<double>(tr, c, e, f);
      const bool found = count_near(res.candidates, e.x, rho) > 0;
      bool in_gap = false;
      for (const auto& g : res.gaps) {
        in_gap = in_gap || (g.first - rho <= e.x && e.x <= g.second + rho);
      }
      CAPTURE(e.x);
      CHECK((found || in_gap));
    }
  }
}

// ---- Validation --------------------------------------------------------------------------------

TEST_CASE_TEMPLATE("roots validation: empty coefficients and non-positive tolerances throw", T,
                   float, double) {
  const pr::Builtin b;
  const std::vector<T> c = {T(1), T(-1)};
  const std::span<const T> good(c);
  const std::span<const T> empty;
  CHECK_THROWS_WITH_AS(static_cast<void>(b.find_roots<T>(empty, {1e-10, 1e-10})),
                       "coeff must have at least 1 element", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(b.find_roots<T>(good, {0.0, 1e-10})),
                       doctest::Contains("param_tol must be positive"), pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(b.find_roots<T>(good, {-1e-10, 1e-10})),
                       doctest::Contains("param_tol must be positive"), pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(b.find_roots<T>(good, {1e-10, 0.0})),
                       doctest::Contains("value_tol must be positive"), pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(b.find_roots<T>(good, {1e-10, -1.0})),
                       doctest::Contains("value_tol must be positive"), pantr::value_error);
  // The monotone entry points and the batch ones validate alike.
  CHECK_THROWS_AS(static_cast<void>(b.find_monotone_root<T>(empty, 1e-10)), pantr::value_error);
  CHECK_THROWS_AS(static_cast<void>(b.find_monotone_root<T>(good, 0.0)), pantr::value_error);
  const std::vector<T> flat = {T(1), T(-1), T(1), T(-1)};
  const pantr::mdview<const T, 2> no_cols(flat.data(), 4, 0);
  const pantr::mdview<const T, 2> two_cols(flat.data(), 2, 2);
  CHECK_THROWS_AS(static_cast<void>(b.find_roots_batch<T>(no_cols, {1e-10, 1e-10})),
                  pantr::value_error);
  CHECK_THROWS_AS(static_cast<void>(b.find_roots_batch<T>(two_cols, {0.0, 1e-10})),
                  pantr::value_error);
  CHECK_THROWS_AS(static_cast<void>(b.find_roots_batch<T>(two_cols, {1e-10, 0.0})),
                  pantr::value_error);
  CHECK_THROWS_AS(static_cast<void>(b.find_monotone_roots_batch<T>(no_cols, 1e-10)),
                  pantr::value_error);
  CHECK_THROWS_AS(static_cast<void>(b.find_monotone_roots_batch<T>(two_cols, 0.0)),
                  pantr::value_error);
}

TEST_CASE_TEMPLATE("roots entry point: a Bezier must be scalar and one-dimensional", T, float,
                   double) {
  const std::vector<T> surface(9, T(1));
  const pantr::bezier::Bezier<T> two_d(std::span<const T>(surface), {3, 3, 1}, false);
  CHECK_THROWS_WITH_AS(static_cast<void>(pr::find_roots<pr::Builtin, T>(two_d, {1e-10, 1e-10})),
                       "Bezier must be 1-D (dim == 1), got dim=2", pantr::value_error);
  const std::vector<T> planar(6, T(1));
  const pantr::bezier::Bezier<T> curve2(std::span<const T>(planar), {3, 2}, false);
  CHECK_THROWS_WITH_AS(static_cast<void>(pr::find_roots<pr::Builtin, T>(curve2, {1e-10, 1e-10})),
                       "Bezier must be scalar (rank == 1), got rank=2", pantr::value_error);
  // A rational planar curve has rank 2 as well (cp_size 3, the weight excluded).
  const std::vector<T> rational_planar(9, T(1));
  const pantr::bezier::Bezier<T> curve3(std::span<const T>(rational_planar), {3, 3}, true);
  CHECK_THROWS_WITH_AS(static_cast<void>(pr::find_roots<pr::Builtin, T>(curve3, {1e-10, 1e-10})),
                       "Bezier must be scalar (rank == 1), got rank=2", pantr::value_error);
  // The dimension is checked first: a surface with a vector value reports dim.
  const std::vector<T> vec_surface(18, T(1));
  const pantr::bezier::Bezier<T> both(std::span<const T>(vec_surface), {3, 3, 2}, false);
  CHECK_THROWS_WITH_AS(static_cast<void>(pr::find_roots<pr::Builtin, T>(both, {1e-10, 1e-10})),
                       doctest::Contains("dim == 1"), pantr::value_error);
}

TEST_CASE_TEMPLATE("roots entry point: a rational scalar curve is solved on its first column", T,
                   float, double) {
  // Bug caught: the weight column (or the quotient) used instead of the first homogeneous
  // component, or a stride error that mixes the columns. The numerator is the encoded-truth
  // polynomial of degree 7; the weights are positive and unlike it, so a wrong column finds no root
  // or the wrong ones.
  const Truth tr = make_truth({{rat(2, 16), 1},
                               {rat(3, 16), 1},
                               {rat(5, 16), 1},
                               {rat(8, 16), 1},
                               {rat(11, 16), 1},
                               {rat(13, 16), 1},
                               {rat(14, 16), 1}});
  const std::vector<T> numerator = stored<T>(tr);
  const std::vector<double> weights = {1.0, 2.0, 0.5, 3.0, 1.5, 0.75, 2.5, 1.25};
  std::vector<T> homogeneous;
  for (std::size_t i = 0; i < numerator.size(); ++i) {
    homogeneous.push_back(numerator[i]);
    homogeneous.push_back(static_cast<T>(weights[i]));
  }
  const pantr::bezier::Bezier<T> rational(std::span<const T>(homogeneous), {8, 2}, true);
  const pantr::bezier::Bezier<T> plain(std::span<const T>(numerator), {8, 1}, false);
  REQUIRE(rational.rank() == 1);
  CHECK(pr::scalar_coefficients<T>(rational) == numerator);
  const Frame f = frame_for<T>(tr, numerator);
  const pr::RootSet<double> from_rational =
      pr::find_roots<pr::Builtin, T>(rational, {f.param_tol, f.value_tol});
  const pr::RootSet<double> from_plain =
      pr::find_roots<pr::Builtin, T>(plain, {f.param_tol, f.value_tol});
  check_roots_match<T>(from_rational, tr, numerator, f);
  REQUIRE(from_rational.roots.size() == from_plain.roots.size());
  for (std::size_t i = 0; i < from_plain.roots.size(); ++i) {
    CHECK(from_rational.roots[i].x == from_plain.roots[i].x);  // the same call on the same data
  }
}

TEST_CASE("roots entry point: the Bezier overload hands a provider the scalar coefficients") {
  // The seam: any RootProvider, not only Builtin, receives the stored coefficients (the first
  // homogeneous column of a rational curve) and the options as given. Bug caught: the entry point
  // binding to Builtin, or dropping the options.
  const std::vector<double> homogeneous = {0.5, 1.0, -2.0, 4.0, 3.0, 2.0};
  const pantr::bezier::Bezier<double> rational(std::span<const double>(homogeneous), {3, 2}, true);
  const RecordingProvider provider;
  const pr::RootSet<double> rs = pr::find_roots(rational, {0.125, 0.25}, provider);
  CHECK(provider.seen == std::vector<double>{0.5, -2.0, 3.0});
  CHECK(provider.options.param_tol == 0.125);
  CHECK(provider.options.value_tol == 0.25);
  REQUIRE(rs.roots.size() == 1);
  CHECK(rs.roots[0].claim == pr::Claim::certified);
  CHECK(rs.roots[0].multiplicity == 1);
}

// ---- The Newton cap ----------------------------------------------------------------------------

TEST_CASE_TEMPLATE("roots newton cap: an exhausted hybrid returns the midpoint of its bracket", T,
                   float, double) {
  // p(t) = (t - 4/5)(t + 1) = t^2 + t/5 - 4/5, increasing on [0, 1] with Bernstein coefficients
  // (-4/5, -7/10, 2/5) (b_1 = b_0 + a_1 / 2). Hand-computed iterates, analytic oracle:
  //   * 0 iterations: the bracket is [0, 1], so the midpoint is 1/2;
  //   * 1 iteration of solve_monotone: false position x0 = (4/5) / (6/5) = 2/3 (inside the clamp
  //     [0.1, 0.9]), p(2/3) < 0 so the bracket is [2/3, 1] and the midpoint is 5/6;
  //   * 1 iteration of solve_on_interval from the midpoint x = 1/2: p(1/2) = -9/20 < 0, bracket
  //     [1/2, 1], midpoint 3/4;
  //   * the default cap of 64 reaches the root 4/5.
  // Bug caught: the cap returning NaN, the last iterate, or the root without ever bracketing it;
  // an off-by-one in the loop count.
  const Truth tr = make_truth({{rat(4, 5), 1}, {rat(-1), 1}});
  const std::vector<T> c = stored<T>(tr);
  const double u = unit_roundoff<T>();
  const std::span<const T> s(c);
  CHECK(prd::solve_monotone<T>(s, 1e-12, 0) == 0.5);
  CHECK(prd::solve_on_interval<T>(s, 0.0, 1.0, static_cast<double>(c[0]), 1e-12, 0) == 0.5);
  CHECK(std::fabs(prd::solve_monotone<T>(s, 1e-12, 1) - 5.0 / 6.0) <= 16.0 * u);
  CHECK(std::fabs(prd::solve_on_interval<T>(s, 0.0, 1.0, static_cast<double>(c[0]), 1e-12, 1) -
                  0.75) <= 16.0 * u);
  const Frame f = frame_for<T>(tr, c);
  const double rho = position_bound<T>(tr, c, tr.in_unit_interval[0], f);
  CHECK(std::fabs(prd::solve_monotone<T>(s, f.param_tol) - 0.8) <= rho);
  CHECK(std::fabs(prd::solve_on_interval<T>(s, 0.0, 1.0, static_cast<double>(c[0]), f.param_tol) -
                  0.8) <= rho);
}

// ---- The monotone solver -----------------------------------------------------------------------

TEST_CASE_TEMPLATE(
    "roots monotone: a monotone polynomial's root is found, or NaN without a sign "
    "change",
    T, float, double) {
  // Encoded truth: (t - 1/3)(t + 2)^3 is increasing on [0, 1] (its derivative is (t + 2)^2 (4t +
  // 1) > 0), its negative decreasing. Bug caught: a bracket update with the wrong sign
  // convention for a decreasing polynomial; a finite number returned where the end values do not
  // change sign.
  const pr::Builtin b;
  for (const long long sign : {1LL, -1LL}) {
    const Truth tr = make_truth({{rat(1, 3), 1}, {rat(-2), 3}}, rat(sign));
    const std::vector<T> c = stored<T>(tr);
    const Frame f = frame_for<T>(tr, c);
    const double rho = position_bound<T>(tr, c, tr.in_unit_interval[0], f);
    CAPTURE(sign);
    CHECK(std::fabs(b.find_monotone_root<T>(std::span<const T>(c), f.param_tol) - 1.0 / 3.0) <=
          rho);
  }
  // A root at an end of the interval is found within param_tol (the bracket halves onto it).
  const std::vector<T> at_zero = {T(0), T(1), T(2)};
  const std::vector<T> at_one = {T(-2), T(-1), T(0)};
  CHECK(std::fabs(b.find_monotone_root<T>(std::span<const T>(at_zero), 1e-10)) <= 1e-10);
  CHECK(std::fabs(b.find_monotone_root<T>(std::span<const T>(at_one), 1e-10) - 1.0) <= 1e-10);
  // No sign change between the end values: NaN, for a positive polynomial and for one that dips
  // below zero in between (the solver's precondition is monotone, so it does not look inside).
  const std::vector<T> positive = {T(1), T(1.5), T(2)};
  const std::vector<T> negative = {T(-1), T(-1.5), T(-2)};
  const std::vector<T> dip = {T(1), T(-5), T(1)};
  CHECK(std::isnan(b.find_monotone_root<T>(std::span<const T>(positive), 1e-10)));
  CHECK(std::isnan(b.find_monotone_root<T>(std::span<const T>(negative), 1e-10)));
  CHECK(std::isnan(b.find_monotone_root<T>(std::span<const T>(dip), 1e-10)));
}

TEST_CASE_TEMPLATE("roots monotone: batch rows agree with single calls, NaN for no root", T, float,
                   double) {
  const pr::Builtin b;
  const std::vector<std::vector<T>> rows = {
      stored<T>(make_truth({{rat(1, 3), 1}, {rat(-2), 3}})),
      stored<T>(make_truth({{rat(1, 3), 1}, {rat(-2), 3}}, rat(-1))),
      {T(1), T(1.5), T(2), T(2.5), T(3)},
      {T(0), T(1), T(2), T(3), T(4)},
      stored<T>(make_truth({{rat(7, 8), 1}, {rat(-1), 3}})),
  };
  pantr::NdArray<T> m({rows.size(), rows[0].size()});
  for (std::size_t i = 0; i < rows.size(); ++i) {
    std::copy(rows[i].begin(), rows[i].end(), m.data() + i * rows[0].size());
  }
  const std::vector<double> got = b.find_monotone_roots_batch<T>(
      pantr::mdview<const T, 2>(m.data(), rows.size(), rows[0].size()), 1e-10);
  REQUIRE(got.size() == rows.size());
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const double single = b.find_monotone_root<T>(std::span<const T>(rows[i]), 1e-10);
    CAPTURE(i);
    if (std::isnan(single)) {
      CHECK(std::isnan(got[i]));
    } else {
      CHECK(std::fabs(got[i] - single) <= 1e-10);
    }
  }
  CHECK(std::isnan(got[2]));  // the positive row has no root: the batch says so too
}

// ---- The batch ---------------------------------------------------------------------------------

namespace {

template <class T>
pantr::NdArray<T> stack_rows(const std::vector<std::vector<T>>& rows) {
  pantr::NdArray<T> m({rows.size(), rows.front().size()});
  for (std::size_t i = 0; i < rows.size(); ++i) {
    std::copy(rows[i].begin(), rows[i].end(), m.data() + i * rows.front().size());
  }
  return m;
}

}  // namespace

TEST_CASE_TEMPLATE(
    "roots batch: rows agree with single calls on the clipping path and pad with "
    "NaN",
    T, float, double) {
  // Both dispatches clip and merge at degree >= 6 within the coefficient range, so a row of the
  // batch is the single call's result. Bug caught: a row offset (a stride error across rows), a
  // count that disagrees with the roots written, a missing NaN padding, a completeness vector of
  // the wrong length, or a batch that does not merge. 5 distinct rows cycled over 300 polynomials
  // so that the work is split over the pool's chunks (a race on `counts` or `roots` shows under
  // TSan).
  const std::vector<std::vector<Factor>> row_factors = {
      {{rat(2, 16), 1},
       {rat(3, 16), 1},
       {rat(5, 16), 1},
       {rat(8, 16), 1},
       {rat(11, 16), 1},
       {rat(13, 16), 1},
       {rat(14, 16), 1}},  // 7 roots: a full row
      {{rat(1, 4), 1},
       {rat(1, 2), 1},
       {rat(3, 4), 1},
       {rat(-1), 1},
       {rat(2), 1},
       {rat(3), 1},
       {rat(-2), 1}},  // 3 roots, 4 NaN
      {{rat(-1), 1},
       {rat(2), 1},
       {rat(3), 1},
       {rat(-2), 1},
       {rat(4), 1},
       {rat(-3), 1},
       {rat(5), 1}},  // none
      {{rat(0), 1},
       {rat(1), 1},
       {rat(1, 2), 1},
       {rat(-1), 1},
       {rat(2), 1},
       {rat(3), 1},
       {rat(-2), 1}},  // 0, 1/2, 1
      {{rat(3, 8), 1},
       {rat(-1), 1},
       {rat(2), 1},
       {rat(3), 1},
       {rat(-2), 1},
       {rat(4), 1},
       {rat(-3), 1}},  // 1 root
  };
  std::vector<Truth> truths;
  std::vector<std::vector<T>> distinct;
  double value_tol = 0.0;
  for (const auto& f : row_factors) {
    truths.push_back(unit_scale_truth(f));
    distinct.push_back(stored<T>(truths.back()));
    value_tol = (std::max)(value_tol, frame_for<T>(truths.back(), distinct.back()).value_tol);
  }
  std::vector<std::vector<T>> rows(300);
  std::vector<std::size_t> which(rows.size());
  for (std::size_t i = 0; i < rows.size(); ++i) {
    which[i] = (i * 7 + i / 5) % distinct.size();
    rows[i] = distinct[which[i]];
  }
  rows[17] = std::vector<T>(8, T(0));  // the zero polynomial: no roots, whatever the tolerance
  const Frame f{1e-10, value_tol};
  const pantr::NdArray<T> m = stack_rows(rows);
  const pr::BatchRootSet br = pr::Builtin{}.find_roots_batch<T>(
      pantr::mdview<const T, 2>(m.data(), rows.size(), 8), {f.param_tol, f.value_tol});
  REQUIRE(br.roots.shape()[0] == rows.size());
  REQUIRE(br.roots.shape()[1] == 7);  // max(degree, 1)
  REQUIRE(br.counts.size() == rows.size());
  REQUIRE(br.completeness.size() == rows.size());
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const std::size_t w = br.roots.shape()[1];
    const pr::RootSet<double> single = builtin_roots<T>(rows[i], f);
    CAPTURE(i);
    const std::size_t expected_count = i == 17 ? 0 : truths[which[i]].in_unit_interval.size();
    REQUIRE(br.counts[i] == static_cast<std::int64_t>(expected_count));
    REQUIRE(single.roots.size() == expected_count);
    CHECK(br.completeness[i] == pr::Completeness::complete);
    for (std::size_t j = 0; j < w; ++j) {
      const double v = br.roots.data()[i * w + j];
      if (j < expected_count) {
        REQUIRE_FALSE(std::isnan(v));
        CHECK(std::fabs(v - single.roots[j].x) <= f.param_tol);
        if (j > 0) {
          CHECK(v > br.roots.data()[i * w + j - 1]);
        }
      } else {
        CHECK(std::isnan(v));  // padding, and nothing but padding, after `counts` roots
      }
    }
    if (i != 17) {
      const Truth& tr = truths[which[i]];
      for (std::size_t j = 0; j < expected_count; ++j) {
        const double rho = position_bound<T>(tr, distinct[which[i]], tr.in_unit_interval[j], f);
        CHECK(std::fabs(br.roots.data()[i * w + j] - tr.in_unit_interval[j].x) <= rho);
      }
    }
  }
}

TEST_CASE_TEMPLATE("roots batch: Yuksel rows with simple roots agree with single calls", T, float,
                   double) {
  const std::vector<std::vector<Factor>> row_factors = {
      {{rat(1, 4), 1}, {rat(3, 4), 1}},
      {{rat(1, 8), 1}, {rat(1, 2), 1}, {rat(7, 8), 1}},
      {{rat(-1), 1}, {rat(2), 1}, {rat(3), 1}},
      {{rat(0), 1}, {rat(1, 2), 1}, {rat(1), 1}},
  };
  std::vector<std::vector<T>> rows;
  std::vector<Truth> truths;
  for (const auto& f : row_factors) {
    // Pad every row to degree 3 with roots outside [0, 1] (the batch has one degree).
    std::vector<Factor> padded = f;
    for (int pad = 0; padded.size() < 3; ++pad) {
      padded.push_back({rat(2 + pad), 1});
    }
    truths.push_back(make_truth(padded));
    rows.push_back(stored<T>(truths.back()));
  }
  const pantr::NdArray<T> m = stack_rows(rows);
  const pr::BatchRootSet br = pr::Builtin{}.find_roots_batch<T>(
      pantr::mdview<const T, 2>(m.data(), rows.size(), 4), {1e-10, 1e-10});
  REQUIRE(br.roots.shape()[1] == 3);
  for (std::size_t i = 0; i < rows.size(); ++i) {
    const pr::RootSet<double> single = builtin_roots<T>(rows[i], {1e-10, 1e-10});
    CAPTURE(i);
    REQUIRE(br.counts[i] == static_cast<std::int64_t>(single.roots.size()));
    REQUIRE(br.counts[i] == static_cast<std::int64_t>(truths[i].in_unit_interval.size()));
    for (std::size_t j = 0; j < single.roots.size(); ++j) {
      CHECK(std::fabs(br.roots.data()[i * 3 + j] - single.roots[j].x) <= 1e-10);
    }
  }
}

TEST_CASE("roots batch: degrees 0 and 1 give one column, and empty batches are empty") {
  const pr::Builtin b;
  const std::vector<double> constants = {2.0, 0.0, -1.0};
  const pr::BatchRootSet c0 = b.find_roots_batch<double>(
      pantr::mdview<const double, 2>(constants.data(), 3, 1), {1e-10, 1e-10});
  CHECK(c0.roots.shape()[0] == 3);
  CHECK(c0.roots.shape()[1] == 1);  // max(degree, 1)
  for (std::size_t i = 0; i < 3; ++i) {
    CHECK(c0.counts[i] == 0);  // a nonzero constant and the zero polynomial have no root to report
    CHECK(std::isnan(c0.roots.data()[i]));
  }
  const std::vector<double> lines = {-1.0, 3.0, 2.0, 1.0, 1.0, -1.0};
  const pr::BatchRootSet c1 = b.find_roots_batch<double>(
      pantr::mdview<const double, 2>(lines.data(), 3, 2), {1e-10, 1e-10});
  REQUIRE(c1.roots.shape()[1] == 1);
  CHECK(c1.counts[0] == 1);
  CHECK(std::fabs(c1.roots.data()[0] - 0.25) <= 1e-10);  // (-1, 3): root at 1/4
  CHECK(c1.counts[1] == 0);                              // (2, 1): no sign change
  CHECK(std::isnan(c1.roots.data()[1]));
  CHECK(c1.counts[2] == 1);
  CHECK(std::fabs(c1.roots.data()[2] - 0.5) <= 1e-10);  // (1, -1): root at 1/2
  const pr::BatchRootSet none = b.find_roots_batch<double>(
      pantr::mdview<const double, 2>(lines.data(), 0, 2), {1e-10, 1e-10});
  CHECK(none.roots.shape()[0] == 0);
  CHECK(none.counts.empty());
  CHECK(none.completeness.empty());
}
