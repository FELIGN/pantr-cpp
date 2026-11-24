// Knot insertion, conversion to open and periodic form, subdivision, restriction, splitting and the
// blossom of a B-spline.
//
// Oracles: Boehm's one-knot insertion (Piegl and Tiller A5.1) applied repeatedly in exact rational
// arithmetic (every knot and control point is a float or a double, hence a dyadic rational, so the
// reference carries no rounding), which is the definition of knot insertion and not the discrete
// B-spline recurrence the library runs; the exact-rational Cox-de Boor sum of the control points
// (bspline_oracle.hpp) for "the curve is preserved"; and, for the blossom, its defining
// properties (diagonal equals the curve, symmetry, multi-affinity), the curve being the exact sum.
// Tolerances are derived where they are used from the unit roundoff of the storage type.
#include <doctest/doctest.h>

#include "bspline_oracle.hpp"

#include <pantr/bezier/detail/along_axis.hpp>
#include <pantr/bspline/blossom.hpp>
#include <pantr/bspline/bspline.hpp>
#include <pantr/bspline/evaluate.hpp>
#include <pantr/bspline/knot_factories.hpp>
#include <pantr/bspline/knot_insertion.hpp>
#include <pantr/bspline/periodic.hpp>
#include <pantr/bspline/restrict.hpp>
#include <pantr/core/format.hpp>
#include <pantr/tolerance.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace bs = pantr::bspline;
namespace oracle = bspline_oracle;
using oracle::Rational;
using pantr::mdview;
using pantr::NdArray;

namespace {

template <class T>
double unit_roundoff() {
  return 0.5 * static_cast<double>(std::numeric_limits<T>::epsilon());
}

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
std::vector<double> values(std::size_t count, unsigned seed) {
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

/// A knot vector with `ends` copies of 0 and of `hi` and the given interior knots.
std::vector<double> clamped(int degree, const std::vector<double>& interior, double hi) {
  std::vector<double> k(static_cast<std::size_t>(degree) + 1, 0.0);
  k.insert(k.end(), interior.begin(), interior.end());
  k.insert(k.end(), static_cast<std::size_t>(degree) + 1, hi);
  return k;
}

// ---- the exact reference: Boehm's one-knot insertion --------------------------------------------

struct ExactNet {
  std::vector<Rational> u;
  int p;
  std::vector<std::vector<Rational>> P;  // one row of coordinates per control point
};

ExactNet exact_net(const std::vector<double>& knots, int p, const std::vector<double>& flat,
                   std::size_t cols) {
  ExactNet net{{}, p, {}};
  for (const double k : knots) {
    net.u.push_back(oracle::exact(k));
  }
  for (std::size_t i = 0; i + cols <= flat.size(); i += cols) {
    std::vector<Rational> row;
    for (std::size_t c = 0; c < cols; ++c) {
      row.push_back(oracle::exact(flat[i + c]));
    }
    net.P.push_back(std::move(row));
  }
  return net;
}

/// Insert `t` once (Piegl and Tiller A5.1 with r = 1): `u_k <= t < u_{k+1}` for the last such `k`
/// in
/// `[p, n]`, `s` the multiplicity of `t` at `k`.
ExactNet boehm_insert(const ExactNet& in, const Rational& t) {
  const auto p = static_cast<std::size_t>(in.p);
  const std::size_t n = in.P.size() - 1;
  std::size_t k = p;
  for (std::size_t j = p; j <= n; ++j) {
    if (in.u[j] <= t) {
      k = j;
    }
  }
  std::size_t s = 0;
  for (std::size_t j = 0; j <= k; ++j) {
    if (in.u[j] == t) {
      ++s;
    }
  }
  ExactNet out;
  out.p = in.p;
  out.u.assign(in.u.begin(), in.u.begin() + static_cast<std::ptrdiff_t>(k) + 1);
  out.u.push_back(t);
  out.u.insert(out.u.end(), in.u.begin() + static_cast<std::ptrdiff_t>(k) + 1, in.u.end());
  const std::size_t cols = in.P[0].size();
  out.P.assign(n + 2, std::vector<Rational>(cols, Rational(0)));
  for (std::size_t i = 0; i + p <= k; ++i) {
    out.P[i] = in.P[i];
  }
  for (std::size_t i = k - s; i <= n; ++i) {
    out.P[i + 1] = in.P[i];
  }
  for (std::size_t i = k - p + 1; i <= k - s; ++i) {
    const Rational alpha = (t - in.u[i]) / (in.u[i + p] - in.u[i]);
    for (std::size_t c = 0; c < cols; ++c) {
      out.P[i][c] = alpha * in.P[i][c] + (1 - alpha) * in.P[i - 1][c];
    }
  }
  return out;
}

ExactNet boehm_insert_all(ExactNet net, const std::vector<double>& knots) {
  for (const double t : knots) {
    net = boehm_insert(net, oracle::exact(t));
  }
  return net;
}

/// The exact value of the curve `sum_i P_i N_i(x)` of a 1D net (the points taken as stored when
/// `wrap` is nonzero: `P[i mod wrap]`).
std::vector<Rational> exact_curve(const ExactNet& net, const Rational& x, std::size_t wrap = 0) {
  oracle::BasisAt at(net.u, net.p, x);
  const std::size_t cols = net.P[0].size();
  std::vector<Rational> out(cols, Rational(0));
  const std::size_t first = at.span() - static_cast<std::size_t>(net.p);
  for (std::size_t j = 0; j <= static_cast<std::size_t>(net.p); ++j) {
    const std::size_t i = first + j;
    const Rational w = at.deriv(i, 0);
    if (w != 0) {
      const auto& row = net.P[wrap == 0 ? i : i % wrap];
      for (std::size_t c = 0; c < cols; ++c) {
        out[c] += w * row[c];
      }
    }
  }
  return out;
}

template <class T>
bs::BsplineSpace1D<T> space_1d(const std::vector<double>& knots, int p, bool periodic = false) {
  const std::vector<T> k = cast_all<T>(knots);
  return bs::BsplineSpace1D<T>(sp(k), p, periodic);
}

template <class T>
bs::Bspline<T> curve(const std::vector<double>& knots, int p, const std::vector<double>& flat,
                     bool periodic = false) {
  std::vector<bs::BsplineSpace1D<T>> spaces;
  spaces.push_back(space_1d<T>(knots, p, periodic));
  const std::vector<T> cp = cast_all<T>(flat);
  return bs::Bspline<T>(bs::BsplineSpace<T>(std::move(spaces)), sp(cp), false);
}

/// The largest |got - want| over a flat array against exact rationals, as a double.
template <class T>
double max_error(std::span<const T> got, const std::vector<std::vector<Rational>>& want) {
  double worst = 0.0;
  std::size_t at = 0;
  for (const auto& row : want) {
    for (const Rational& w : row) {
      Rational d = oracle::exact(static_cast<double>(got[at++])) - w;
      if (d < 0) {
        d = -d;
      }
      worst = (std::max)(worst, static_cast<double>(d));
    }
  }
  return worst;
}

struct Case1D {
  std::vector<double> knots;
  int p;
  std::vector<double> insert;
};

// Clamped and unclamped vectors; interior and repeated insertions; a knot added at a domain end of
// the unclamped one (multiplicity one there).
const std::vector<Case1D> kCases = {
    {clamped(2, {0.25, 0.75, 0.75}, 1.0), 2, {0.125, 0.5, 0.75}},
    {clamped(3, {0.25, 0.5, 0.75}, 1.0), 3, {0.375, 0.375, 0.875, 0.5}},
    {clamped(1, {0.5}, 1.0), 1, {0.25, 0.75}},
    {clamped(4, {0.5}, 1.0), 4, {0.25, 0.25, 0.25, 0.75}},
    {{-1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5}, 2, {0.25, 1.0, 0.25}},
    {{-1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5}, 2, {0.0, 1.5}},
    {clamped(0, {0.5}, 1.0), 0, {0.25, 0.75}},
};

}  // namespace

TEST_CASE_TEMPLATE("knot insertion: the Oslo matrix equals repeated Boehm insertion", T, double,
                   float) {
  // Oracle: Boehm's insertion applied to the columns of the identity, in exact rationals. Each
  // entry of a row is a convex combination built by p levels of three roundings (subtract, divide,
  // multiply) and one addition on nonnegative terms, so it is within 4 p u of the exact entry
  // (the carried sum is formed in double for float, which only tightens it); 6 p u keeps a margin
  // of 50 percent for contraction and vector width.
  for (const Case1D& c : kCases) {
    const std::vector<T> old_knots = cast_all<T>(c.knots);
    const std::size_t n1 = old_knots.size() - static_cast<std::size_t>(c.p) - 1;
    std::vector<double> eye(n1 * n1, 0.0);
    for (std::size_t i = 0; i < n1; ++i) {
      eye[i * n1 + i] = 1.0;
    }
    const ExactNet refined = boehm_insert_all(exact_net(c.knots, c.p, eye, n1), c.insert);
    const std::vector<T> merged = bs::compute_inserted_knot_vector<T>(
        sp(old_knots), c.p, sp(cast_all<T>(c.insert)), bs::knot_tolerance<T>(sp(old_knots)));
    const NdArray<T> oslo = bs::oslo_matrix<T>(c.p, sp(old_knots), sp(merged));
    REQUIRE(oslo.shape()[0] == refined.P.size());
    REQUIRE(oslo.shape()[1] == n1);
    const double tol = 6.0 * static_cast<double>(c.p) * unit_roundoff<T>();
    CHECK(max_error<T>(oslo.flat(), refined.P) <= tol);
  }
}

TEST_CASE_TEMPLATE("knot insertion: the refined control points equal repeated Boehm insertion", T,
                   double, float) {
  // The rows' entries are within 6 p u (above), the dot product of p + 1 terms adds (p + 1) u of
  // the sum of magnitudes: 8 (p + 1) u max|P| covers both with the same margin.
  for (const Case1D& c : kCases) {
    const std::size_t n1 = c.knots.size() - static_cast<std::size_t>(c.p) - 1;
    const std::size_t cols = 2;
    const std::vector<double> flat = values(n1 * cols, 7);
    const std::vector<T> knots = cast_all<T>(c.knots);
    const std::vector<T> cp = cast_all<T>(flat);
    const std::vector<T> add = cast_all<T>(c.insert);
    const double tol_k = bs::knot_tolerance<T>(sp(knots));
    const bs::KnotsAndCtrl<T> got = bs::insert_knots_1d<T>(
        sp(knots), c.p, mdview<const T, 2>(cp.data(), n1, cols), sp(add), tol_k);
    const ExactNet want = boehm_insert_all(exact_net(c.knots, c.p, flat, cols), c.insert);
    REQUIRE(got.ctrl.shape()[0] == want.P.size());
    const double tol =
        8.0 * (static_cast<double>(c.p) + 1.0) * unit_roundoff<T>() * 1.0 /* max |P| <= 1 */;
    CHECK(max_error<T>(got.ctrl.flat(), want.P) <= tol);
    // The knot vector is the sorted union.
    REQUIRE(got.knots.size() == want.u.size());
    for (std::size_t i = 0; i < want.u.size(); ++i) {
      CHECK(oracle::exact(static_cast<double>(got.knots[i])) == want.u[i]);
    }
  }
}

TEST_CASE_TEMPLATE("knot insertion: the refined curve equals the original at other parameters", T,
                   double, float) {
  // Oracle: the exact Cox-de Boor sum on the original and the refined knot vector; the two exact
  // sums differ by the rounding of the refined control points only, 8 (p + 1) u max|P| as above.
  for (const Case1D& c : kCases) {
    if (c.p == 0) {
      continue;
    }
    const std::size_t n1 = c.knots.size() - static_cast<std::size_t>(c.p) - 1;
    const std::vector<double> flat = values(n1 * 3, 11);
    const bs::Bspline<T> before = curve<T>(c.knots, c.p, flat);
    const std::vector<T> add = cast_all<T>(c.insert);
    const bs::Bspline<T> after =
        bs::insert_knots<T>(before, std::span<const std::span<const T>>(
                                        std::vector<std::span<const T>>{std::span<const T>(add)}));
    const auto sp_after = after.space().spaces()[0];
    const auto sp_before = before.space().spaces()[0];
    const std::vector<double> after_knots(sp_after.knots().begin(), sp_after.knots().end());
    const std::vector<double> before_knots(sp_before.knots().begin(), sp_before.knots().end());
    const std::vector<double> after_cp(after.control_points().begin(),
                                       after.control_points().end());
    const ExactNet a = exact_net(before_knots, c.p, flat, 3);
    const ExactNet b = exact_net(after_knots, c.p, after_cp, 3);
    const auto lo = static_cast<double>(sp_before.domain().first);
    const auto hi = static_cast<double>(sp_before.domain().second);
    const double tol = 8.0 * (static_cast<double>(c.p) + 1.0) * unit_roundoff<T>();
    for (int j = 0; j <= 16; ++j) {
      const double x = lo + (hi - lo) * static_cast<double>(j) / 16.0 * 0.999;
      const Rational xr = oracle::exact(x);
      const auto va = exact_curve(a, xr);
      const auto vb = exact_curve(b, xr);
      for (std::size_t k = 0; k < 3; ++k) {
        Rational d = va[k] - vb[k];
        if (d < 0) {
          d = -d;
        }
        CHECK(static_cast<double>(d) <= tol);
      }
    }
  }
}

TEST_CASE("knot insertion: refusals and their messages") {
  const std::vector<double> k = clamped(2, {0.5}, 1.0);
  const std::vector<double> cp = values((k.size() - 3) * 1, 3);
  const bs::Bspline<double> b = curve<double>(k, 2, cp);
  const auto call = [&](std::vector<double> add) {
    const std::span<const double> s(add);
    return bs::insert_knots<double>(b, std::vector<std::span<const double>>{s});
  };
  const auto message_of = [&](const std::vector<double>& add) -> std::string {
    try {
      static_cast<void>(call(add));
    } catch (const pantr::value_error& e) {
      return e.what();
    }
    return {};
  };
  CHECK(message_of({}) == "At least one direction must have a non-empty array of knots to insert.");
  CHECK(message_of({1.5, -0.25})
            .starts_with("new_knots contains values outside the domain [0.0, 1.0]: ["));
  CHECK(message_of({0.5, 0.5, 0.5})
            .starts_with("Inserting these knots would exceed the maximum "
                         "multiplicity of 3. Maximum multiplicity found: 4."));
  // Two more copies of the one at 0.5 make three: allowed.
  CHECK(message_of({0.5, 0.5}).empty());
  // A knot count that does not match the dimension.
  CHECK_THROWS_WITH_AS(
      (static_cast<void>(bs::insert_knots<double>(b, std::vector<std::span<const double>>{}))),
      "new_knots sequence length (0) must match dim (1).", pantr::value_error);
  // The control points must match the knots, in the 1D entry point.
  const std::vector<double> few(3, 0.0);
  const std::vector<double> add{0.25};
  CHECK_THROWS_AS(static_cast<void>(bs::insert_knots_1d<double>(
                      sp(k), 2, mdview<const double, 2>(few.data(), 3, 1), sp(add), 1e-12)),
                  pantr::value_error);
  CHECK_THROWS_AS(static_cast<void>(bs::compute_inserted_knot_vector<double>(
                      sp(k), 2, std::span<const double>(), 1e-12)),
                  pantr::value_error);
  // The Oslo matrix refuses only what would read out of bounds.
  CHECK_THROWS_AS(static_cast<void>(bs::oslo_matrix<double>(-1, sp(k), sp(k))), pantr::value_error);
  const std::vector<double> tiny{0.0, 1.0};
  CHECK_THROWS_AS(static_cast<void>(bs::oslo_matrix<double>(2, sp(tiny), sp(k))),
                  pantr::value_error);
}

TEST_CASE_TEMPLATE("knot insertion: the largest multiplicity agrees with the knot classes", T,
                   double, float) {
  // The merge test counts runs itself (no allocation); it must group as the classes do.
  const std::vector<std::vector<double>> vectors = {
      {0, 0, 0, 0.5, 0.5, 1, 1, 1},
      {0, 0, 0, 0.25, 0.5, 0.75, 1, 1, 1},
      {0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 1, 1, 1},
      {-1, -0.5, 0, 0.5, 1, 1.5, 2},
  };
  for (const auto& v : vectors) {
    const std::vector<T> k = cast_all<T>(v);
    const double tol = bs::knot_tolerance<T>(sp(k));
    const bs::KnotClasses<T> c = bs::unique_knots_and_multiplicity<T>(sp(k), 1, tol, false);
    const std::int64_t want = *std::max_element(c.multiplicities.begin(), c.multiplicities.end());
    CHECK(bs::detail::max_multiplicity<T>(sp(k), tol) == want);
  }
  CHECK(bs::detail::max_multiplicity<T>(std::span<const T>(), 1e-12) == 0);
}

TEST_CASE_TEMPLATE("knot insertion: a multiplicity past degree + 1 is refused at every degree", T,
                   double, float) {
  for (int p = 1; p <= 4; ++p) {
    const std::vector<double> k = clamped(p, {0.5}, 1.0);
    const std::vector<T> kt = cast_all<T>(k);
    const std::vector<T> add(static_cast<std::size_t>(p) + 1, static_cast<T>(0.5));
    // One more than the existing single copy: p + 2 copies in all.
    CHECK_THROWS_AS(static_cast<void>(bs::compute_inserted_knot_vector<T>(
                        sp(kt), p, sp(add), bs::knot_tolerance<T>(sp(kt)))),
                    pantr::value_error);
    const std::vector<T> ok(static_cast<std::size_t>(p), static_cast<T>(0.5));
    CHECK_NOTHROW(static_cast<void>(
        bs::compute_inserted_knot_vector<T>(sp(kt), p, sp(ok), bs::knot_tolerance<T>(sp(kt)))));
  }
}

TEST_CASE_TEMPLATE("to_open: a periodic and an unclamped direction give the same curve", T, double,
                   float) {
  // Periodic: uniform C^1 quadratic, 5 intervals; the exact curve of the periodic space is the sum
  // over the full (ghost-extended) knot vector with the stored points wrapped.
  {
    const std::vector<T> kt =
        bs::uniform_periodic_knots<T>(5, 2, 1, static_cast<T>(0), static_cast<T>(1));
    const std::vector<double> knots(kt.begin(), kt.end());
    const bs::BsplineSpace1D<T> ps(sp(kt), 2, true);
    const auto n_stored = static_cast<std::size_t>(ps.num_basis());
    const std::vector<double> flat = values(n_stored * 2, 5);
    const bs::Bspline<T> b = curve<T>(knots, 2, flat, true);
    const bs::Bspline<T> open = bs::to_open_bspline<T>(b);
    CHECK_FALSE(open.space().spaces()[0].periodic());
    CHECK(open.space().spaces()[0].has_open_knots());
    const ExactNet per = exact_net(knots, 2, flat, 2);
    const auto os = open.space().spaces()[0];
    const std::vector<double> ok(os.knots().begin(), os.knots().end());
    const std::vector<double> ocp(open.control_points().begin(), open.control_points().end());
    const ExactNet on = exact_net(ok, 2, ocp, 2);
    const double tol = 8.0 * 3.0 * unit_roundoff<T>();
    for (int j = 0; j <= 20; ++j) {
      const Rational x = oracle::exact(static_cast<double>(j) / 20.0 * 0.999);
      const auto vp = exact_curve(per, x, n_stored);
      const auto vo = exact_curve(on, x);
      for (std::size_t k = 0; k < 2; ++k) {
        Rational d = vp[k] - vo[k];
        if (d < 0) {
          d = -d;
        }
        CHECK(static_cast<double>(d) <= tol);
      }
    }
  }
  // Unclamped, non-periodic.
  {
    const std::vector<double> knots = {-1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5};
    const std::vector<double> flat = values((knots.size() - 3) * 2, 9);
    const bs::Bspline<T> b = curve<T>(knots, 2, flat);
    const bs::Bspline<T> open = bs::to_open_bspline<T>(b);
    CHECK(open.space().spaces()[0].has_open_knots());
    const auto os = open.space().spaces()[0];
    CHECK(static_cast<double>(os.domain().first) == 0.0);
    CHECK(static_cast<double>(os.domain().second) == 1.5);
    const std::vector<double> ok(os.knots().begin(), os.knots().end());
    const std::vector<double> ocp(open.control_points().begin(), open.control_points().end());
    const ExactNet a = exact_net(knots, 2, flat, 2);
    const ExactNet on = exact_net(ok, 2, ocp, 2);
    const double tol = 8.0 * 3.0 * unit_roundoff<T>();
    for (int j = 0; j <= 20; ++j) {
      const Rational x = oracle::exact(static_cast<double>(j) / 20.0 * 1.4);
      const auto va = exact_curve(a, x);
      const auto vo = exact_curve(on, x);
      for (std::size_t k = 0; k < 2; ++k) {
        Rational d = va[k] - vo[k];
        if (d < 0) {
          d = -d;
        }
        CHECK(static_cast<double>(d) <= tol);
      }
    }
  }
}

TEST_CASE("to_open: an open B-spline is refused, direction by direction") {
  const std::vector<double> k = clamped(2, {0.5}, 1.0);
  const bs::Bspline<double> b = curve<double>(k, 2, values(4, 1));
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::to_open_bspline<double>(b)),
                       "B-spline is already open in every direction.", pantr::value_error);
  const std::vector<double> cp = values(4, 1);
  CHECK_THROWS_WITH_AS(
      (static_cast<void>(bs::to_open_1d<double>(sp(k), 2, mdview<const double, 2>(cp.data(), 4, 1),
                                                false, 1e-12))),
      "B-spline is already open (both boundary knots have multiplicity degree + 1).",
      pantr::value_error);
}

TEST_CASE_TEMPLATE("to_periodic: the round trip of a periodic spline returns its control points", T,
                   double, float) {
  // The conservative tier (4096 eps) bounds the closure residual of the backward-stable QR solve
  // by max|x|; the solution error adds the condition number of M, a few units for uniform spaces
  // (covered: the tier pays for a condition number up to about 1e3).
  for (const int p : {2, 3, 4}) {
    for (const int continuity : {p - 1, 1}) {
      if (continuity > p - 1 || continuity < 0) {
        continue;
      }
      const long long n_int = 2 * p + 3;
      const std::vector<T> kt =
          bs::uniform_periodic_knots<T>(n_int, p, continuity, static_cast<T>(0), static_cast<T>(1));
      const std::vector<double> knots(kt.begin(), kt.end());
      const bs::BsplineSpace1D<T> ps(sp(kt), p, true);
      const auto n_stored = static_cast<std::size_t>(ps.num_basis());
      const std::vector<double> flat = values(n_stored * 2, 21 + static_cast<unsigned>(p));
      const bs::Bspline<T> per = curve<T>(knots, p, flat, true);
      const bs::Bspline<T> open = bs::to_open_bspline<T>(per);
      const std::vector<std::optional<long long>> cont{static_cast<long long>(continuity)};
      const bs::Bspline<T> back = bs::to_periodic<T>(open, cont);
      REQUIRE(back.space().spaces()[0].periodic());
      REQUIRE(back.control_points().size() == flat.size());
      const double tol = static_cast<double>(pantr::tolerance::get_conservative<T>());
      double worst = 0.0;
      for (std::size_t i = 0; i < flat.size(); ++i) {
        worst =
            (std::max)(worst, std::abs(static_cast<double>(back.control_points()[i]) - flat[i]));
      }
      CHECK(worst <= tol);
      // The in-domain knots are the original ones (the factory's ghost knots are spaced singly
      // while the conversion extends the tile periodically, so only the domain is compared).
      const auto bk = back.space().spaces()[0].knots();
      REQUIRE(bk.size() == knots.size());
      for (std::size_t i = static_cast<std::size_t>(p);
           i + static_cast<std::size_t>(p) < knots.size(); ++i) {
        CHECK(static_cast<double>(bk[i]) == doctest::Approx(knots[i]).epsilon(1e-6));
      }
    }
  }
}

TEST_CASE_TEMPLATE("to_periodic: ghost knots outside a clamped domain take the second Oslo step", T,
                   double, float) {
  // The input is a B-spline whose knot vector is the clamped one of a periodic spline with `p`
  // extra knots outside the domain on each side: boundary multiplicity `p + 1`, but not open
  // (`knots[0] != knots[p]`). It has more knots than the periodic vector the conversion builds, so
  // the change of basis is the product of two Oslo steps (the trimmed intermediate to the input
  // knots) and not one. The existing round trips pass through `to_open_bspline` and so never
  // reach it.
  //
  // Oracle, independent of both steps, for everything the domain sees. The function on the domain
  // is that of the clamped spline `O` (control points `P`), and the stored control points of the
  // periodic spline it came from are the expected result. The coefficients of the `p` basis
  // functions at each end that are wholly outside the domain do not change that function, but the
  // conversion fits all of them, so the input must give them the values it expects. At the right
  // end that is the blossom of the last polynomial piece continued past the domain at the
  // function's `p` inner knots: for `p = 1` it is `P_n`; for `p = 2` the two are `P_n` and
  // `f[b, h] = P_n + (h - b)(P_{n-1} - P_n) / (t_n - b)`, `h` the ghost knot nearest the domain
  // and `t_n` the last knot before `b` (the blossom is affine in each argument and
  // `f[t_n, b] = P_{n-1}`). At the left end it is zero, not the continuation: the conversion
  // treats those functions as absent, as the pure-Python implementation it replaces did (a
  // continuation there is refused with a closure residual, on both).
  struct Layout {
    int degree;
    int continuity;
  };
  for (const Layout lay : {Layout{1, 0}, Layout{2, 0}, Layout{2, 1}}) {
    const int p = lay.degree;
    CAPTURE(p);
    CAPTURE(lay.continuity);
    const long long n_int = 2 * p + 3;
    const std::vector<T> kt = bs::uniform_periodic_knots<T>(n_int, p, lay.continuity,
                                                            static_cast<T>(0), static_cast<T>(1));
    const std::vector<double> knots(kt.begin(), kt.end());
    const bs::BsplineSpace1D<T> ps(sp(kt), p, true);
    const auto n_stored = static_cast<std::size_t>(ps.num_basis());
    const std::vector<double> stored = values(n_stored, 41 + static_cast<unsigned>(p));
    const bs::Bspline<T> open = bs::to_open_bspline<T>(curve<T>(knots, p, stored, true));
    const auto os = open.space().spaces()[0];
    const std::vector<T> ok(os.knots().begin(), os.knots().end());
    const std::vector<double> clamped_pts(open.control_points().begin(),
                                          open.control_points().end());
    const std::size_t n_open = clamped_pts.size();
    const auto up = static_cast<std::size_t>(p);

    // The input: p ghost knots on each side, spaced as the first and last spans.
    const double a = static_cast<double>(ok[up]);
    const double b = static_cast<double>(ok[ok.size() - up - 1]);
    const double tn = static_cast<double>(ok[ok.size() - up - 2]);
    std::vector<T> v;
    for (std::size_t j = 0; j < up; ++j) {
      v.push_back(static_cast<T>(a - 0.25 * static_cast<double>(up - j)));
    }
    v.insert(v.end(), ok.begin(), ok.end());
    for (std::size_t j = 0; j < up; ++j) {
      v.push_back(static_cast<T>(b + 0.25 * static_cast<double>(j + 1)));
    }
    std::vector<double> u_pts(up, 0.0);
    u_pts.insert(u_pts.end(), clamped_pts.begin(), clamped_pts.end());
    u_pts.push_back(clamped_pts[n_open - 1]);
    if (p == 2) {
      const double h = b + 0.25;
      u_pts.push_back(clamped_pts[n_open - 1] +
                      (h - b) * (clamped_pts[n_open - 2] - clamped_pts[n_open - 1]) / (tn - b));
    }
    REQUIRE(u_pts.size() == v.size() - up - 1);
    const std::vector<T> u_cast = cast_all<T>(u_pts);

    const double knot_tol = bs::BsplineSpace1D<T>(sp(v), p).tolerance();
    const bs::KnotsAndCtrl<T> per =
        bs::to_periodic_1d<T>(sp(v), p, mdview<const T, 2>(u_cast.data(), u_cast.size(), 1),
                              p - lay.continuity, knot_tol);
    // The second step ran: the periodic vector the conversion builds is not the input's size.
    CHECK(per.knots.size() != v.size());
    REQUIRE(per.ctrl.shape()[0] == n_stored);

    // The stored control points come back, within the conservative tier the closure budget uses
    // (values are at most 1 in magnitude).
    const double tier = static_cast<double>(pantr::tolerance::get_conservative<T>());
    double worst = 0.0;
    for (std::size_t i = 0; i < n_stored; ++i) {
      worst = (std::max)(worst, std::abs(static_cast<double>(per.ctrl.data()[i]) - stored[i]));
    }
    CHECK(worst <= tier);

    // The periodic result equals the input on the domain: the exact curves, the input over its
    // own knots and the result over its ghost-extended knots with its points wrapped.
    const std::vector<double> vk(v.begin(), v.end());
    const std::vector<double> rk(per.knots.begin(), per.knots.end());
    const std::vector<double> rp(per.ctrl.data(), per.ctrl.data() + n_stored);
    const ExactNet input_net =
        exact_net(vk, p, std::vector<double>(u_cast.begin(), u_cast.end()), 1);
    const ExactNet result_net = exact_net(rk, p, rp, 1);
    const double curve_tol = 8.0 * 3.0 * unit_roundoff<T>() + tier;
    for (int j = 0; j <= 20; ++j) {
      const Rational x = oracle::exact(a + (b - a) * static_cast<double>(j) / 20.0 * 0.999);
      Rational d = exact_curve(input_net, x)[0] - exact_curve(result_net, x, n_stored)[0];
      if (d < 0) {
        d = -d;
      }
      CHECK(static_cast<double>(d) <= curve_tol);
    }
  }
}

TEST_CASE("to_periodic: refusals and their messages") {
  const std::vector<double> k = clamped(2, {0.25, 0.5, 0.75}, 1.0);
  const std::size_t n = k.size() - 3;
  std::vector<double> cp = values(n, 4);
  // The end points differ: the seam test.
  cp[0] = 0.5;
  cp[n - 1] = -0.5;
  const bs::Bspline<double> not_closed = curve<double>(k, 2, cp);
  const std::vector<std::optional<long long>> def{bs::convert_default_continuity};
  try {
    static_cast<void>(bs::to_periodic<double>(not_closed, def));
    FAIL("expected value_error");
  } catch (const pantr::value_error& e) {
    const std::string m = e.what();
    CHECK(m.starts_with(
        "Function values do not match at the domain endpoints (max deviation 1.00e+00, "
        "tolerance "));
    CHECK(m.ends_with("). The B-spline is not periodic."));
  }
  // Closed ends but not smooth across the seam: the residual test.
  cp[n - 1] = cp[0];
  const bs::Bspline<double> kinked = curve<double>(k, 2, cp);
  try {
    static_cast<void>(bs::to_periodic<double>(kinked, def));
    FAIL("expected value_error");
  } catch (const pantr::value_error& e) {
    const std::string m = e.what();
    CHECK(m.starts_with("Residual "));
    CHECK(m.find(" exceeds tolerance ") != std::string::npos);
    CHECK(
        m.ends_with(". The B-spline does not have sufficient smoothness at the seam for the "
                    "requested periodic regularity."));
  }
  // The continuity range, the tuple length, nothing to convert.
  CHECK_THROWS_WITH_AS((static_cast<void>(bs::to_periodic<double>(
                           kinked, std::vector<std::optional<long long>>{2}))),
                       "continuity must be in [0, degree-1]=[0, 1] for direction 0, got 2.",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS((static_cast<void>(bs::to_periodic<double>(
                           kinked, std::vector<std::optional<long long>>{0, 0}))),
                       "continuity tuple length 2 != dimension 1.", pantr::value_error);
  CHECK_THROWS_WITH_AS(
      (static_cast<void>(
          bs::to_periodic<double>(kinked, std::vector<std::optional<long long>>{std::nullopt}))),
      "No direction to convert: all directions are either already periodic or explicitly skipped.",
      pantr::value_error);
  // The boundary multiplicity range of the 1D entry point.
  CHECK_THROWS_AS((bs::check_boundary_multiplicity(0, 2)), pantr::value_error);
  CHECK_THROWS_AS((bs::check_boundary_multiplicity(3, 2)), pantr::value_error);
  CHECK_NOTHROW(bs::check_boundary_multiplicity(2, 2));
  CHECK_THROWS_AS((bs::check_boundary_multiplicity(1, 0)), pantr::value_error);
}

TEST_CASE("to_periodic: the budgets are the conservative tier times the magnitude") {
  const std::vector<double> c{0.5, -2.0, 1.0};
  const double cons = static_cast<double>(pantr::tolerance::get_conservative<double>());
  CHECK(bs::control_point_magnitude<double>(sp(c)) == 2.0);
  CHECK(bs::seam_deviation_budget<double>(sp(c)) == cons * 2.0);
  const std::vector<double> per{4.0, 1.0};
  CHECK(bs::closure_deviation_budget<double>(sp(c), sp(per), std::nullopt) == cons * 4.0);
  // The caller's budget raises the floor, never lowers it.
  CHECK(bs::closure_deviation_budget<double>(sp(c), sp(per), 1e-30) == cons * 4.0);
  CHECK(bs::closure_deviation_budget<double>(sp(c), sp(per), 1e3) == 1e3);
  CHECK(bs::control_point_magnitude<double>(std::span<const double>()) == 0.0);
  const std::vector<double> nan{1.0, std::numeric_limits<double>::quiet_NaN()};
  CHECK(std::isnan(bs::control_point_magnitude<double>(sp(nan))));
  // The float budget reads eps from float.
  const std::vector<float> cf{1.0F};
  CHECK(bs::seam_deviation_budget<float>(sp(cf)) ==
        static_cast<double>(pantr::tolerance::get_conservative<float>()));
}

TEST_CASE("to_periodic: the ghost knots of a vector with fewer entries than ghosts") {
  // A tile of one entry and three ghosts per side: the left loop takes three passes and keeps
  // them in the order they were made (a vector no space accepts; see build_periodic_knot_vector).
  const std::vector<double> open = clamped(3, {}, 1.0);
  const std::vector<double> kv = bs::build_periodic_knot_vector<double>(
      sp(open), 3, std::span<const double>(), std::span<const std::int64_t>(), 1);
  const std::vector<double> want{-1, -2, -3, 0, 1, 2, 3, 4};
  CHECK(kv == want);
  // The ordinary case: the tile covers the ghosts.
  const std::vector<double> open2 = clamped(2, {0.5}, 1.0);
  const std::vector<double> ip{0.5};
  const std::vector<std::int64_t> im{1};
  const std::vector<double> kv2 =
      bs::build_periodic_knot_vector<double>(sp(open2), 2, sp(ip), sp(im), 1);
  // n_ghost = 2 per side: the left ghosts are the tile shifted by -1, the right ones the interior
  // knot and the next tile's leading knot.
  const std::vector<double> want2{-1, -0.5, 0, 0.5, 1, 1.5, 2};
  CHECK(kv2 == want2);
}

TEST_CASE_TEMPLATE("insert_knots: a periodic direction stays periodic and keeps its curve", T,
                   double, float) {
  const std::vector<T> kt =
      bs::uniform_periodic_knots<T>(6, 2, 1, static_cast<T>(0), static_cast<T>(1));
  const std::vector<double> knots(kt.begin(), kt.end());
  const bs::BsplineSpace1D<T> ps(sp(kt), 2, true);
  const auto n_stored = static_cast<std::size_t>(ps.num_basis());
  const std::vector<double> flat = values(n_stored * 2, 17);
  const bs::Bspline<T> per = curve<T>(knots, 2, flat, true);
  const std::vector<T> add = cast_all<T>({0.125, 0.5});
  const bs::Bspline<T> refined =
      bs::insert_knots<T>(per, std::vector<std::span<const T>>{std::span<const T>(add)});
  const auto rs = refined.space().spaces()[0];
  REQUIRE(rs.periodic());
  CHECK(rs.num_basis() == ps.num_basis() + 2);
  const std::vector<double> rk(rs.knots().begin(), rs.knots().end());
  const std::vector<double> rcp(refined.control_points().begin(), refined.control_points().end());
  const ExactNet a = exact_net(knots, 2, flat, 2);
  const ExactNet b = exact_net(rk, 2, rcp, 2);
  const double tol = static_cast<double>(pantr::tolerance::get_conservative<T>());
  for (int j = 0; j <= 24; ++j) {
    const Rational x = oracle::exact(static_cast<double>(j) / 24.0 * 0.999);
    const auto va = exact_curve(a, x, n_stored);
    const auto vb = exact_curve(b, x, static_cast<std::size_t>(rs.num_basis()));
    for (std::size_t k = 0; k < 2; ++k) {
      Rational d = va[k] - vb[k];
      if (d < 0) {
        d = -d;
      }
      CHECK(static_cast<double>(d) <= tol);
    }
  }
}

TEST_CASE("subdivide: the knots, the counts and the refusals") {
  const std::vector<double> k = clamped(3, {0.5}, 1.0);
  const std::vector<double> cp = values((k.size() - 4) * 1, 2);
  const bs::Bspline<double> b = curve<double>(k, 3, cp);
  const bs::Bspline<double> sub = bs::subdivide<double>(b, std::vector<long long>{4});
  // Two intervals of 4 sub-spans at regularity degree - 1: 3 new knots in each.
  CHECK(sub.space().spaces()[0].num_basis() == b.space().spaces()[0].num_basis() + 6);
  const std::vector<double> kk =
      bs::compute_uniform_subdivision_knots<double>(sp(k), 3, 1e-12, 2, 1);
  REQUIRE(kk.size() == 4);  // 2 intervals, 1 value each, repeated degree - regularity = 2 times
  CHECK(kk[0] == 0.25);
  CHECK(kk[1] == 0.25);
  CHECK(kk[2] == 0.75);
  CHECK(kk[3] == 0.75);
  // Messages.
  CHECK_THROWS_WITH_AS((static_cast<void>(bs::subdivide<double>(b, std::vector<long long>{1}))),
                       "At least one direction must have n_subdivisions >= 2.", pantr::value_error);
  CHECK_THROWS_WITH_AS((static_cast<void>(bs::subdivide<double>(b, std::vector<long long>{0}))),
                       "n_subdivisions must be >= 1, got 0", pantr::value_error);
  CHECK_THROWS_WITH_AS((static_cast<void>(bs::subdivide<double>(b, std::vector<long long>{2, 2}))),
                       "n_subdivisions sequence length (2) must match dim (1).",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS((static_cast<void>(bs::subdivide<double>(b, std::vector<long long>{2}, 3))),
                       "regularity must be in [-1, degree - 1] = [-1, 2] for direction 0, got 3",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(
      (static_cast<void>(bs::compute_uniform_subdivision_knots<double>(sp(k), 3, 1e-12, 1))),
      "n_subdivisions must be >= 2, got 1", pantr::value_error);
  // Regularity -1 repeats each knot degree + 1 times.
  const std::vector<double> kk1 =
      bs::compute_uniform_subdivision_knots<double>(sp(k), 3, 1e-12, 2, -1);
  CHECK(kk1.size() == 8);
}

TEST_CASE_TEMPLATE("restrict and split: the pieces carry the curve of the original", T, double,
                   float) {
  const std::vector<double> k = clamped(3, {0.25, 0.5, 0.75}, 1.0);
  const std::size_t n1 = k.size() - 4;
  const std::vector<double> flat = values(n1 * 2, 31);
  const bs::Bspline<T> b = curve<T>(k, 3, flat);
  const ExactNet orig = exact_net(k, 3, flat, 2);
  const double tol = 16.0 * 4.0 * unit_roundoff<T>();

  const auto check_piece = [&](const bs::Bspline<T>& piece, double lo, double hi) {
    const auto s = piece.space().spaces()[0];
    CHECK(static_cast<double>(s.domain().first) == lo);
    CHECK(static_cast<double>(s.domain().second) == hi);
    CHECK(s.has_open_knots());
    const std::vector<double> pk(s.knots().begin(), s.knots().end());
    const std::vector<double> pc(piece.control_points().begin(), piece.control_points().end());
    const ExactNet net = exact_net(pk, 3, pc, 2);
    for (int j = 0; j <= 12; ++j) {
      const Rational x = oracle::exact(lo + (hi - lo) * static_cast<double>(j) / 12.0 * 0.99);
      const auto va = exact_curve(orig, x);
      const auto vb = exact_curve(net, x);
      for (std::size_t c = 0; c < 2; ++c) {
        Rational d = va[c] - vb[c];
        if (d < 0) {
          d = -d;
        }
        CHECK(static_cast<double>(d) <= tol);
      }
    }
  };

  const bs::Bspline<T> mid =
      bs::restrict<T>(b, std::vector<std::optional<bs::Bounds>>{bs::Bounds{0.125, 0.625}});
  check_piece(mid, 0.125, 0.625);
  const bs::Bspline<T> left_only =
      bs::restrict<T>(b, std::vector<std::optional<bs::Bounds>>{bs::Bounds{0.0, 0.5}});
  check_piece(left_only, 0.0, 0.5);

  const auto [l, r] = bs::split<T>(b, 0, 0.375);
  check_piece(l, 0.0, 0.375);
  check_piece(r, 0.375, 1.0);
  // Splitting at an existing knot inserts nothing more than needed.
  const auto [l2, r2] = bs::split<T>(b, 0, 0.5);
  check_piece(l2, 0.0, 0.5);
  check_piece(r2, 0.5, 1.0);
}

TEST_CASE("restrict and split: refusals and their messages") {
  const std::vector<double> k = clamped(2, {0.5}, 1.0);
  const bs::Bspline<double> b = curve<double>(k, 2, values(4, 1));
  const auto one = [&](double lo, double hi) {
    return bs::restrict<double>(b, std::vector<std::optional<bs::Bounds>>{bs::Bounds{lo, hi}});
  };
  CHECK_THROWS_WITH_AS(static_cast<void>(one(0.5, 0.5)),
                       "Lower bound (0.5) must be strictly less than upper bound (0.5).",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(one(-0.5, 0.5)),
                       "Lower bound (-0.5) is below the domain start (0.0).", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(one(0.5, 1.5)),
                       "Upper bound (1.5) is above the domain end (1.0).", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(one(0.0, 1.0)),
                       "Bounds match the full domain and the direction is already open.",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS((static_cast<void>(bs::restrict<double>(
                           b, std::vector<std::optional<bs::Bounds>>{std::nullopt}))),
                       "At least one direction must have non-None bounds that restrict the domain.",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(
      (static_cast<void>(bs::restrict<double>(b, std::vector<std::optional<bs::Bounds>>{}))),
      "bounds sequence length (0) must match dim (1).", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::split<double>(b, 1, 0.5)),
                       "direction must be in [0, 1), got 1.", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::split<double>(b, -1, 0.5)),
                       "direction must be in [0, 1), got -1.", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::split<double>(b, 0, 1.0)),
                       "value must be strictly inside the domain (0.0, 1.0), got 1.0.",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::split<double>(b, 0, 0.0)),
                       "value must be strictly inside the domain (0.0, 1.0), got 0.0.",
                       pantr::value_error);
}

TEST_CASE("knot kernels refuse a degree the knot vector cannot hold, before indexing it") {
  // Six knots, exactly sized: an index past them is a heap overread that AddressSanitizer reports.
  // Degree 9 needs 20 knots, degree -1 has no meaning; both used to index `knots[degree]` and
  // `knots[size - degree - 1]` unchecked.
  const std::vector<double> k{0.0, 0.0, 0.0, 1.0, 1.0, 1.0};
  const std::vector<double> add{0.5};
  const std::vector<double> cp = values(3, 8);
  const mdview<const double, 2> ctrl(cp.data(), 3, 1);
  const std::string too_few = "knots must have at least 2*degree+2 elements";
  const std::string negative = "degree must be non-negative";
  const bs::Bounds bounds{0.25, 0.75};

  const auto refused = [&](auto&& call, const std::string& message) {
    CHECK_THROWS_WITH_AS(static_cast<void>(call()), message.c_str(), pantr::value_error);
  };
  for (const int degree : {9, 3}) {
    // Three is the first degree for which six knots are too few (2 * 3 + 2 = 8).
    refused([&] { return bs::compute_inserted_knot_vector<double>(sp(k), degree, sp(add), 1e-12); },
            too_few);
    refused([&] { return bs::validate_restrict_bounds<double>(sp(k), degree, 1e-12, bounds); },
            too_few);
    refused(
        [&] { return bs::compute_boundary_knots_to_insert<double>(sp(k), degree, 1e-12, bounds); },
        too_few);
  }
  refused([&] { return bs::compute_inserted_knot_vector<double>(sp(k), -1, sp(add), 1e-12); },
          negative);
  refused([&] { return bs::validate_restrict_bounds<double>(sp(k), -1, 1e-12, bounds); }, negative);
  refused([&] { return bs::compute_boundary_knots_to_insert<double>(sp(k), -1, 1e-12, bounds); },
          negative);
  // Exactly 2 * degree + 2 knots is accepted: degree 2 on six knots.
  CHECK_NOTHROW(
      static_cast<void>(bs::compute_inserted_knot_vector<double>(sp(k), 2, sp(add), 1e-12)));
  CHECK_NOTHROW(static_cast<void>(bs::validate_restrict_bounds<double>(sp(k), 2, 1e-12, bounds)));

  // The raw conversions and the subdivision knots indexed before they checked: `to_open_1d` read
  // `knots[SIZE_MAX]` at degree -1, the subdivision dereferenced an empty vector, and
  // `to_periodic_1d` built Oslo matrices of no space on 13 knots at degree 7.
  const std::vector<double> five{0.0, 1.0, 2.0, 3.0, 4.0};
  const std::vector<double> cp7 = values(7, 1);
  const mdview<const double, 2> ctrl7(cp7.data(), 7, 1);
  refused([&] { return bs::to_open_1d<double>(sp(five), -1, ctrl7, true, 1e-12); }, negative);
  refused([&] { return bs::to_open_1d<double>(sp(k), 9, ctrl, true, 1e-12); }, too_few);
  refused([&] { return bs::to_open_1d<double>(sp(k), 9, ctrl, false, 1e-12); }, too_few);
  const std::vector<double> none;
  refused([&] { return bs::compute_uniform_subdivision_knots<double>(sp(none), 2, 1e-12, 2); },
          too_few);
  refused([&] { return bs::compute_uniform_subdivision_knots<double>(sp(k), -1, 1e-12, 2); },
          negative);
  const std::vector<double> thirteen{0.0, 1.0, 2.0, 3.0,  4.0,  5.0, 6.0,
                                     7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
  const std::vector<double> cp5 = values(5, 1);
  const mdview<const double, 2> ctrl5(cp5.data(), 5, 1);
  refused([&] { return bs::to_periodic_1d<double>(sp(thirteen), 7, ctrl5, 1, 1e-12); }, too_few);
  refused([&] { return bs::to_periodic_1d<double>(sp(thirteen), -1, ctrl5, 1, 1e-12); }, negative);

  // The whole-direction entry points, which check the control net first.
  for (const int degree : {9, -1}) {
    const std::string& message = degree < 0 ? negative : too_few;
    // A net that matches no knot vector at that degree is refused as a mismatch (degree 9), and a
    // negative degree is refused outright, in either case before the bounds or the value are read.
    CHECK_THROWS_AS(static_cast<void>(bs::restrict_1d<double>(sp(k), degree, ctrl, false, 1e-12,
                                                              bs::Bounds{0.75, 0.25})),
                    pantr::value_error);
    CHECK_THROWS_AS(static_cast<void>(bs::split_1d<double>(sp(k), degree, ctrl, false, 1e-12, 0.5)),
                    pantr::value_error);
    // A periodic direction skips the net check, so the knot vector's own size is what refuses.
    refused(
        [&] {
          return bs::restrict_1d<double>(sp(k), degree, ctrl, true, 1e-12, bs::Bounds{0.25, 0.5});
        },
        message);
  }
  // The net is checked before the bounds: a mismatched net and bounds out of order report the net.
  const std::vector<double> few(2, 0.0);
  CHECK_THROWS_WITH_AS((static_cast<void>(bs::restrict_1d<double>(
                           sp(k), 2, mdview<const double, 2>(few.data(), 2, 1), false, 1e-12,
                           bs::Bounds{0.75, 0.25}))),
                       "restrict: 2 control-point rows do not match 6 knots at degree 2.",
                       pantr::value_error);
  // The Oslo kernels' sizes.
  CHECK_THROWS_WITH_AS((bs::detail::check_oslo_sizes(-1, 6, 6)), "degree must be non-negative",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS((bs::detail::check_oslo_sizes(2, 3, 6)),
                       "Both knot vectors need at least degree + 2 = 4 knots, got 3 and 6.",
                       pantr::value_error);
  CHECK_NOTHROW(bs::detail::check_oslo_sizes(2, 4, 4));
}

TEST_CASE("restrict and split: a two-direction B-spline keeps its other direction") {
  const std::vector<double> k = clamped(2, {0.5}, 1.0);
  std::vector<bs::BsplineSpace1D<double>> spaces;
  spaces.push_back(space_1d<double>(k, 2));
  spaces.push_back(space_1d<double>(clamped(1, {0.25, 0.75}, 1.0), 1));
  const bs::BsplineSpace<double> space(spaces);
  const std::vector<double> cp = values(4 * 4 * 3, 5);
  const bs::Bspline<double> b(space, sp(cp), false);
  const auto [l, r] = bs::split<double>(b, 1, 0.5);
  CHECK(l.space().spaces()[0].num_basis() == 4);
  CHECK(r.space().spaces()[0].num_basis() == 4);
  CHECK(l.space().spaces()[1].domain().second == 0.5);
  CHECK(r.space().spaces()[1].domain().first == 0.5);
  CHECK(l.shape()[2] == 3);
  const bs::Bspline<double> x = bs::restrict<double>(
      b, std::vector<std::optional<bs::Bounds>>{std::nullopt, bs::Bounds{0.25, 0.75}});
  CHECK(x.space().spaces()[0].num_basis() == 4);
  CHECK(x.space().spaces()[1].domain().first == 0.25);
}

TEST_CASE_TEMPLATE("blossom: the diagonal is the curve, symmetric and multi-affine", T, double,
                   float) {
  // Oracle: the defining properties of the blossom. The curve is the exact Cox-de Boor sum; the
  // diagonal agrees with it to the rounding of the recurrence, 8 (p + 1) u max|P|.
  for (const int p : {1, 2, 3, 4}) {
    const std::vector<double> kd = clamped(p, {0.25, 0.5, 0.75}, 1.0);
    const std::size_t n1 = kd.size() - static_cast<std::size_t>(p) - 1;
    const std::vector<double> flat = values(n1 * 2, 40 + static_cast<unsigned>(p));
    const std::vector<T> kt = cast_all<T>(kd);
    const std::vector<T> cp = cast_all<T>(flat);
    const mdview<const T, 2> ctrl(cp.data(), n1, 2);
    const double tol_k = bs::knot_tolerance<T>(sp(kt));
    const double tol = 8.0 * (static_cast<double>(p) + 1.0) * unit_roundoff<T>();
    const ExactNet net = exact_net(kd, p, flat, 2);
    for (const double t : {0.0, 0.125, 0.3125, 0.5, 0.875, 1.0}) {
      const std::vector<T> diag(static_cast<std::size_t>(p), static_cast<T>(t));
      const std::vector<T> got = bs::blossom<T>(sp(kt), p, ctrl, sp(diag), tol_k);
      const auto want = exact_curve(net, oracle::exact(t));
      for (std::size_t c = 0; c < 2; ++c) {
        Rational d = oracle::exact(static_cast<double>(got[c])) - want[c];
        if (d < 0) {
          d = -d;
        }
        CHECK(static_cast<double>(d) <= tol);
      }
    }
    // Symmetry: reversing the arguments.
    std::vector<T> args = cast_all<T>({0.125, 0.375, 0.5, 0.625});
    args.resize(static_cast<std::size_t>(p));
    std::vector<T> rev(args.rbegin(), args.rend());
    const std::vector<T> f = bs::blossom<T>(sp(kt), p, ctrl, sp(args), tol_k);
    const std::vector<T> g = bs::blossom<T>(sp(kt), p, ctrl, sp(rev), tol_k);
    for (std::size_t c = 0; c < 2; ++c) {
      CHECK(std::abs(static_cast<double>(f[c]) - static_cast<double>(g[c])) <= tol);
    }
    // Multi-affinity in the first argument: f[(1 - s) a + s b, rest] = (1 - s) f[a, rest] + s f[b,
    // rest] for s = 1/2 inside one knot span.
    std::vector<T> aa = args;
    std::vector<T> bb = args;
    std::vector<T> mm = args;
    aa[0] = static_cast<T>(0.0625);
    bb[0] = static_cast<T>(0.1875);
    mm[0] = static_cast<T>(0.125);
    const std::vector<T> fa = bs::blossom<T>(sp(kt), p, ctrl, sp(aa), tol_k);
    const std::vector<T> fb = bs::blossom<T>(sp(kt), p, ctrl, sp(bb), tol_k);
    const std::vector<T> fm = bs::blossom<T>(sp(kt), p, ctrl, sp(mm), tol_k);
    for (std::size_t c = 0; c < 2; ++c) {
      const double mid = 0.5 * (static_cast<double>(fa[c]) + static_cast<double>(fb[c]));
      CHECK(std::abs(static_cast<double>(fm[c]) - mid) <= tol);
    }
  }
}

TEST_CASE("blossom: refusals, degree zero and the span search") {
  const std::vector<double> k = clamped(2, {0.5}, 1.0);
  const std::vector<double> cp = values(4, 6);
  const mdview<const double, 2> ctrl(cp.data(), 4, 1);
  const std::vector<double> one{0.5};
  const std::vector<double> one2{0.5, 0.5};
  CHECK_THROWS_WITH_AS((static_cast<void>(bs::blossom<double>(sp(k), 2, ctrl, sp(one), 1e-12))),
                       "u_values must have length degree=2, got 1.", pantr::value_error);
  const std::vector<double> out{0.5, 1.5};
  CHECK_THROWS_WITH_AS((static_cast<void>(bs::blossom<double>(sp(k), 2, ctrl, sp(out), 1e-12))),
                       "u_values entry 1.5 is outside domain [0.0, 1.0] (tol=1e-12).",
                       pantr::value_error);
  // Degree zero: the last control point, whatever the (empty) arguments.
  const std::vector<double> k0 = clamped(0, {0.5}, 1.0);
  const std::vector<double> cp0 = values(2, 6);
  const std::vector<double> r0 = bs::blossom<double>(
      sp(k0), 0, mdview<const double, 2>(cp0.data(), 2, 1), std::span<const double>(), 1e-12);
  REQUIRE(r0.size() == 1);
  CHECK(r0[0] == cp0[1]);
  // Fewer knots than a space has: refused, not read past.
  const std::vector<double> short_knots{0.0, 0.0, 1.0, 1.0};
  CHECK_THROWS_WITH_AS(
      (static_cast<void>(bs::blossom<double>(
          sp(short_knots), 2, mdview<const double, 2>(cp.data(), 1, 1), sp(one2), 1e-12))),
      "knots must have at least 2*degree+2 elements", pantr::value_error);
  // A non-clamped vector and a value inside the tolerance below its domain start: the first span,
  // not a window that reaches before the first control point.
  const std::vector<double> kn_open{0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  const std::vector<double> cp_open{1.0, 2.0, 3.0, 4.0};
  const std::vector<double> edge{2.0 - 1e-13, 2.0 - 1e-13};
  const std::vector<double> inside{2.0, 2.0};
  const std::vector<double> got_edge = bs::blossom<double>(
      sp(kn_open), 2, mdview<const double, 2>(cp_open.data(), 4, 1), sp(edge), 1e-9);
  const std::vector<double> got_in = bs::blossom<double>(
      sp(kn_open), 2, mdview<const double, 2>(cp_open.data(), 4, 1), sp(inside), 1e-9);
  // The two values are the curve at parameters 1e-13 apart, so they differ by at most that shift
  // times the largest slope, plus rounding. Slope: the derivative of a degree-p B-spline is
  // p (P_i - P_{i-1}) / (t_{i+p} - t_i) on its pieces; here p = 2, consecutive points differ by 1
  // and the knots are 1 apart, so it is at most 2 * 1 / 2 = 1. Rounding: the recurrence forms two
  // levels of convex combinations of points of magnitude at most 4, taken as 8 * 4 units of
  // roundoff.
  constexpr double shift = 1e-13;
  constexpr double max_slope = 1.0;
  CHECK(std::abs(got_edge[0] - got_in[0]) <=
        max_slope * shift + 8.0 * 4.0 * unit_roundoff<double>());
  // The span of the largest value: the last knot not above it, the rightmost span past the end.
  const std::vector<double> kn{0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0};
  CHECK(bs::detail::blossom_span<double>(sp(kn), 0.25, 3) == 2);
  CHECK(bs::detail::blossom_span<double>(sp(kn), 0.5, 3) == 3);
  CHECK(bs::detail::blossom_span<double>(sp(kn), 1.0, 3) == 3);
  CHECK(bs::detail::blossom_span<double>(sp(kn), std::numeric_limits<double>::quiet_NaN(), 3) == 3);
  // Below the first knot: the span of the first knot itself, the one 0.0 gets (the regression
  // cases below).
}

// ---- below a clamped start (regression) -------------------------------------------------------

TEST_CASE("inherited: blossom span below a clamped start") {
  // The first span of a clamped vector holds the parameters below its start (the blossom of a
  // value within the tolerance below the domain start must be that of the start), which is the
  // span `degree`, here 2. The search used to return the last span.
  const std::vector<double> kn{0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0};
  CHECK(bs::detail::blossom_span<double>(sp(kn), -8.9e-16, 3) == 2);
}

TEST_CASE("inherited: blossom value below a clamped start") {
  // Control points 1..4 of a clamped quadratic: a parameter 4 eps below the start (accepted with a
  // tolerance of 8 eps) must give the value at the start, 1, up to the shift times the slope plus
  // rounding. Slope at most p * max|dP| / min span = 2 * 1 / 0.5 = 4; rounding 8 * 4 units of
  // roundoff for points of magnitude at most 4. The last span's polynomial, which the search used
  // to pick, gives 2.
  const std::vector<double> kn{0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0};
  const std::vector<double> cp{1.0, 2.0, 3.0, 4.0};
  const double eps = std::numeric_limits<double>::epsilon();
  const double u = -8.9e-16;
  const std::vector<double> at_start = {0.0, 0.0};
  const std::vector<double> below = {u, u};
  const mdview<const double, 2> ctrl(cp.data(), 4, 1);
  const std::vector<double> want = bs::blossom<double>(sp(kn), 2, ctrl, sp(at_start), 8.0 * eps);
  const std::vector<double> got = bs::blossom<double>(sp(kn), 2, ctrl, sp(below), 8.0 * eps);
  CHECK(want[0] == 1.0);
  CHECK(std::abs(got[0] - want[0]) <= 4.0 * std::abs(u) + 8.0 * 4.0 * unit_roundoff<double>());
}

TEST_CASE("blossom: a parameter at an end beside an empty span takes the last non-empty span") {
  // Regression. Knots [0, 0, 0, 1, 2, 2, 2, 3, 3] at degree 2: the domain ends at 2, and the last
  // span the search reached there, n = 5, is the empty [2, 2). The blossom now starts from the
  // last non-empty span [1, 2), index 3. Oracle (de Boor on that span by hand, control points
  // P_i = i + 1): f[1.5, 2] takes alpha 0.5 and 0.75 at level 1 (u = 1.5) and 1 at level 2
  // (u = 2), so it is (P2 + P3) / 2 = 3.5; f[2, 2] = P3 = 4, the end value. Rounding: two levels
  // of convex combinations of values at most 6, taken as 8 * 6 units of roundoff.
  const std::vector<double> kn{0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0};
  CHECK(bs::detail::blossom_span<double>(sp(kn), 2.0, 5) == 3);
  CHECK(bs::detail::blossom_span<double>(sp(kn), 1.5, 5) == 3);
  const std::vector<double> cp{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  const mdview<const double, 2> ctrl(cp.data(), 6, 1);
  const double tol = 8.0 * std::numeric_limits<double>::epsilon();
  const double r = 8.0 * 6.0 * unit_roundoff<double>();
  const std::vector<double> mixed = {1.5, 2.0};
  const std::vector<double> end = {2.0, 2.0};
  CHECK(std::abs(bs::blossom<double>(sp(kn), 2, ctrl, sp(mixed), tol)[0] - 3.5) <= r);
  CHECK(std::abs(bs::blossom<double>(sp(kn), 2, ctrl, sp(end), tol)[0] - 4.0) <= r);
}

// ---- a float32 bound within the tolerance of a knot
// -----------------------------------------------------
//
// The previous release counted such a bound as the knot (a `float` difference) and cut the refined
// vector as if it were not (a `double` one), and refused with "knots must have at least
// 2*degree+2 elements". The bound is now snapped onto the knot once, up front.

namespace {

/// A float quadratic on [0, 1] with one interior knot at 0.5, and a bound 2^-20 + 2e-8 above it.
/// The space's tolerance is 8 eps(float) = 2^-20 (span 1), so in `float` the bound rounds to
/// 0.5 + 2^-20, which is within the tolerance of the knot, while in `double` it is 2e-8 beyond it.
constexpr double kNearKnot = 0.5 + 0x1p-20 + 2e-8;

bs::Bspline<float> near_knot_curve() {
  return curve<float>(clamped(2, {0.5}, 1.0), 2, {1.0, 2.0, 3.0, 4.0});
}

/// The knot vector of direction `d`, copied.
std::vector<float> knots_of(const bs::Bspline<float>& b, std::size_t d) {
  const bs::BsplineSpace<float> s = b.space();
  const auto k = s.spaces()[d].knots();
  return {k.begin(), k.end()};
}

/// The control points, copied flat.
std::vector<float> points_of(const bs::Bspline<float>& b) {
  return {b.control_points().begin(), b.control_points().end()};
}

}  // namespace

TEST_CASE("restrict and split: the near-knot float32 bound is the knot by one test only") {
  // The control of the cases below: the two comparisons disagree on the bound, and double knots
  // succeed.
  const bs::Bspline<float> bf = near_knot_curve();
  CHECK(bf.space().spaces()[0].tolerance() == 0x1p-20);
  CHECK(static_cast<double>(static_cast<float>(kNearKnot) - 0.5F) <=
        bf.space().spaces()[0].tolerance());
  CHECK(kNearKnot - 0.5 > bf.space().spaces()[0].tolerance());
  const bs::Bspline<double> bd = curve<double>(clamped(2, {0.5}, 1.0), 2, {1.0, 2.0, 3.0, 4.0});
  CHECK_NOTHROW(static_cast<void>(bs::restrict<double>(
      bd, std::vector<std::optional<bs::Bounds>>{bs::Bounds{kNearKnot, 0.9}})));
  CHECK_NOTHROW(static_cast<void>(bs::split<double>(bd, 0, kNearKnot)));
}

TEST_CASE("inherited: restrict with a float32 bound near a knot") {
  // The restriction at a bound within the tolerance of the knot 0.5 (by the float difference) is
  // the restriction at the knot: same inputs to the same code, so equal knots and points.
  const bs::Bspline<float> bf = near_knot_curve();
  using B = std::vector<std::optional<bs::Bounds>>;
  const bs::Bspline<float> by_bound = bs::restrict<float>(bf, B{bs::Bounds{kNearKnot, 0.9}});
  const bs::Bspline<float> at = bs::restrict<float>(bf, B{bs::Bounds{0.5, 0.9}});
  CHECK(knots_of(by_bound, 0) == knots_of(at, 0));
  CHECK(points_of(by_bound) == points_of(at));
}

TEST_CASE("inherited: split of a float32 spline at a value near a knot") {
  const bs::Bspline<float> bf = near_knot_curve();
  const auto by_bound = bs::split<float>(bf, 0, kNearKnot);
  const auto at = bs::split<float>(bf, 0, 0.5);
  CHECK(knots_of(by_bound.first, 0) == knots_of(at.first, 0));
  CHECK(points_of(by_bound.first) == points_of(at.first));
  CHECK(knots_of(by_bound.second, 0) == knots_of(at.second, 0));
  CHECK(points_of(by_bound.second) == points_of(at.second));
}

TEST_CASE("split: a float32 value that is a domain end is refused as one") {
  // 1 - 2^-20 - 7e-9 is more than the tolerance 2^-20 inside the end 1 in double, and rounds in
  // float to 1 - 2^-20, within it: it is the end, by the comparison that decides which knot a value
  // is, and the split must refuse it as not strictly inside rather than cut an empty part.
  const bs::Bspline<float> bf = near_knot_curve();
  const double near_end = 1.0 - 0x1p-20 - 7e-9;
  REQUIRE(near_end < 1.0 - bf.space().spaces()[0].tolerance());
  REQUIRE(static_cast<double>(1.0F - static_cast<float>(near_end)) <=
          bf.space().spaces()[0].tolerance());
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::split<float>(bf, 0, near_end)),
                       doctest::Contains("value must be strictly inside the domain"),
                       pantr::value_error);
  // The message names the end the value was taken as, since the value itself is inside.
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::split<float>(bf, 0, near_end)),
                       doctest::Contains("of the end knot 1.0."), pantr::value_error);
}

TEST_CASE("restrict: a bound within the tolerance of two knots is the nearer, the first on a tie") {
  // Float knots 0.5 and 0.5 + 1.5 tol. The bound 0.5 + 0.9 tol rounds to 0.5 + 0.875 tol, within
  // the tolerance of both and nearer the second; 0.5 + 0.6 tol rounds to 0.5 + 0.625 tol, nearer
  // the first. 0.5 + 0.75 tol is exact in float and equally far from both (12 ulp): a tie, which
  // goes to the first knot. Each restriction starts with a clamped copy of the knot chosen.
  const double tol = 0x1p-20;
  const double second = static_cast<double>(static_cast<float>(0.5 + 1.5 * tol));
  const std::vector<float> k = {0, 0, 0, 0.5F, static_cast<float>(second), 1, 1, 1};
  std::vector<bs::BsplineSpace1D<float>> spaces;
  spaces.emplace_back(sp(k), 2, false, false);
  const std::vector<float> cp = {1, 2, 3, 4, 5};
  const bs::Bspline<float> b(bs::BsplineSpace<float>(std::move(spaces)), sp(cp), false);
  REQUIRE(b.space().spaces()[0].tolerance() == tol);
  REQUIRE(second - 0.5 == 1.5 * tol);
  for (const auto& [factor, want] :
       {std::pair{0.9, second}, std::pair{0.6, 0.5}, std::pair{0.75, 0.5}}) {
    CAPTURE(factor);
    const double bound = 0.5 + factor * tol;
    const auto rounded = static_cast<double>(static_cast<float>(bound));
    REQUIRE(rounded - 0.5 <= tol);
    REQUIRE(second - rounded <= tol);
    const bs::Bspline<float> r =
        bs::restrict<float>(b, std::vector<std::optional<bs::Bounds>>{bs::Bounds{bound, 0.9}});
    const std::vector<float> got = knots_of(r, 0);
    REQUIRE(got.size() >= 3);
    CHECK(static_cast<double>(got[0]) == want);
    CHECK(static_cast<double>(got[2]) == want);
  }
}

// Refining an axis in place of the axis must give, value for value, what refining the axis moved to
// the front gives (`flatten_along_axis`, `insert_knots_1d`, `unflatten_along_axis`, the route the
// periodic branch still takes): the same bands, applied in the same order. It is compared on a 3D
// net along each axis, for every per-point width from one coordinate to five (one to four run a
// body with the width fixed at compile time, five the run-time one), with a negative zero among
// the control points. Values are compared, not bits: `0 + (-0.0)` is `+0.0` and an assigned first
// term keeps `-0.0`, a difference of sign that compares equal.
TEST_CASE_TEMPLATE(
    "knot insertion: refining an axis in place equals refining it moved to the front", T, double,
    float) {
  const std::vector<T> k0 = {0, 0, 0, T(0.5), 1, 1, 1};
  const std::vector<T> k1 = {0, 0, 1, 1};
  const std::vector<T> k2 = {0, 0, 0, 0, 1, 1, 1, 1};
  const bs::BsplineSpace1D<T> s0(std::span<const T>(k0), 2);
  const bs::BsplineSpace1D<T> s1(std::span<const T>(k1), 1);
  const bs::BsplineSpace1D<T> s2(std::span<const T>(k2), 3);
  const bs::BsplineSpace<T> space(std::vector<bs::BsplineSpace1D<T>>{s0, s1, s2});
  const std::vector<T> ins0 = {T(0.25), T(0.5), T(0.75)};
  const std::vector<T> ins1 = {T(0.125), T(0.375), T(0.625), T(0.875)};
  const std::vector<T> ins2 = {T(0.5), T(0.5)};
  const std::vector<std::vector<T>> inserted = {ins0, ins1, ins2};
  for (std::size_t width = 1; width <= 5; ++width) {
    const std::size_t total = 4 * 2 * 4 * width;
    std::vector<T> cp(total);
    for (std::size_t i = 0; i < total; ++i) {
      cp[i] = static_cast<T>(static_cast<double>((i * 37) % 23) / 7.0 - 1.5);
    }
    cp[1] = T(-0.0);
    const bs::Bspline<T> b(space, std::span<const T>(cp), false);
    for (std::size_t axis = 0; axis < 3; ++axis) {
      CAPTURE(width);
      CAPTURE(axis);
      std::vector<std::span<const T>> per(3);
      per[axis] = inserted[axis];
      const bs::Bspline<T> got =
          bs::insert_knots<T>(b, std::span<const std::span<const T>>(per.data(), per.size()));

      const std::vector<std::size_t> shape(b.shape().begin(), b.shape().end());
      const NdArray<T> flat =
          pantr::bezier::detail::flatten_along_axis<T>(b.control_points(), shape, axis);
      const bs::BsplineSpace1D<T>& dir = space.spaces()[axis];
      const bs::KnotsAndCtrl<T> ref = bs::insert_knots_1d<T>(
          dir.knots(), dir.degree(), flat.template view<2>(), per[axis], dir.tolerance());
      const NdArray<T> want =
          pantr::bezier::detail::unflatten_along_axis<T>(ref.ctrl.template view<2>(), shape, axis);
      REQUIRE(got.size() == want.size());
      const std::span<const T> g = got.control_points();
      for (std::size_t i = 0; i < want.size(); ++i) {
        REQUIRE(g[i] == want.data()[i]);
      }
      const std::span<const T> gk = got.space().spaces()[axis].knots();
      REQUIRE(std::equal(gk.begin(), gk.end(), ref.knots.begin(), ref.knots.end()));
    }
  }
}

// ---- the knots at a restriction or split bound are decided by one rule
// ----------------------------------------------------------------------
//
// The count of the bound's multiplicity and the cut of the refined vector used to classify the
// knots beside the bound separately (a storage-type difference against `tol`, a `double`
// `lower_bound` on `at -+ tol`), and where they disagreed the result was not clamped at the bound
// or had a wrong domain. The four cases are those of the issue, verbatim; each result must have
// `degree + 1` bitwise-equal knots at both ends and the requested domain, or refuse where a
// distinct knot of an unsnapped space within the tolerance of the bound would end up inside it.

namespace {

/// One direction on `knots` at `degree`, snapped or not, control points `i^2 + 1`.
template <class T>
bs::Bspline<T> squares_curve(const std::vector<double>& knots, int degree, bool snap) {
  const std::vector<T> k = cast_all<T>(knots);
  std::vector<bs::BsplineSpace1D<T>> spaces;
  spaces.emplace_back(sp(k), degree, false, snap);
  const auto n = static_cast<std::size_t>(spaces[0].num_basis());
  std::vector<T> cp(n);
  for (std::size_t i = 0; i < n; ++i) {
    cp[i] = static_cast<T>(static_cast<double>(i * i) + 1.0);
  }
  return bs::Bspline<T>(bs::BsplineSpace<T>(std::move(spaces)), sp(cp), false);
}

/// Check the first and the last `degree + 1` knots are bitwise equal, exactly that many, and the
/// domain is `[lo, hi]` exactly.
template <class T>
void check_clamped_on(const bs::Bspline<T>& b, double lo, double hi) {
  const bs::BsplineSpace<T> s = b.space();
  const auto k = s.spaces()[0].knots();
  const auto p = static_cast<std::size_t>(s.spaces()[0].degree());
  REQUIRE(k.size() >= 2 * p + 2);
  const std::size_t n = k.size();
  for (std::size_t i = 0; i <= p; ++i) {
    CAPTURE(i);
    CHECK(k[i] == k[0]);
    CHECK(k[n - 1 - i] == k[n - 1]);
  }
  CHECK(k[p + 1] != k[0]);
  CHECK(k[n - p - 2] != k[n - 1]);
  CHECK(static_cast<double>(k[0]) == lo);
  CHECK(static_cast<double>(k[n - 1]) == hi);
}

using BoundsList = std::vector<std::optional<bs::Bounds>>;

}  // namespace

TEST_CASE("restrict and split: over-collection, a left key that rounds onto a knot (double)") {
  // fl(at - tol) == 0.3 for at = 0.3 + 26 ulp, which the count does not take as the knot.
  const bs::Bspline<double> b = squares_curve<double>({0, 0, 0, 0.3, 0.8, 0.8, 0.8}, 2, true);
  const double ulp = std::nextafter(0.3, 1.0) - 0.3;
  for (const int k : {26, 27}) {
    CAPTURE(k);
    const double at = 0.3 + k * ulp;
    check_clamped_on(bs::restrict<double>(b, BoundsList{bs::Bounds{at, 0.8}}), at, 0.8);
    const auto [left, right] = bs::split<double>(b, 0, at);
    check_clamped_on(left, 0.0, at);
    check_clamped_on(right, at, 0.8);
  }
}

TEST_CASE("restrict and split: over-collection, a float bound within tol of a knot in double") {
  const auto s = static_cast<double>(1.2375F);
  const bs::Bspline<float> b = squares_curve<float>({0, 0, 0, 0.5, s, s, s}, 2, true);
  const double tol = b.space().spaces()[0].tolerance();
  for (const double f : {1.0, 2.0}) {
    CAPTURE(f);
    const double bound = 0.5 + f * tol;
    const auto at = static_cast<double>(static_cast<float>(bound));
    check_clamped_on(bs::restrict<float>(b, BoundsList{bs::Bounds{bound, s}}), at, s);
    const auto [left, right] = bs::split<float>(b, 0, bound);
    check_clamped_on(left, 0.0, at);
    check_clamped_on(right, at, s);
  }
}

TEST_CASE("restrict and split: under-collection, an unsnapped knot exactly tol above the bound") {
  const double eps = std::numeric_limits<double>::epsilon();
  for (const double m : {32.0, 34.0}) {
    CAPTURE(m);
    const bs::Bspline<double> b =
        squares_curve<double>({0, 0, 0, 1.0, 2.0, 2.0 + m * eps, 4.0, 4.0, 4.0}, 2, false);
    check_clamped_on(bs::restrict<double>(b, BoundsList{bs::Bounds{0.0, 2.0}}), 0.0, 2.0);
  }
  // At 32 eps the knot above 2 is within the tolerance of the bound: a piece starting at 2 would
  // hold it beside its clamped start, and the call refuses, naming the bound.
  const bs::Bspline<double> b =
      squares_curve<double>({0, 0, 0, 1.0, 2.0, 2.0 + 32 * eps, 4.0, 4.0, 4.0}, 2, false);
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::split<double>(b, 0, 2.0)),
                       doctest::Contains("The bound 2.0 is within the tolerance"),
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::restrict<double>(b, BoundsList{bs::Bounds{2.0, 4.0}})),
                       doctest::Contains("The bound 2.0 is within the tolerance"),
                       pantr::value_error);
}

TEST_CASE("restrict and split: under-collection, a float count that rounds a knot onto the bound") {
  const double k3 = -(0x1p-20 + 0x1p-43);
  const bs::Bspline<float> b =
      squares_curve<float>({-1, -1, -1, k3, 0x1p-20, 0.5, 0.75, 1, 1, 1}, 2, false);
  check_clamped_on(bs::restrict<float>(b, BoundsList{bs::Bounds{0x1p-20, 1.0}}), 0x1p-20, 1.0);
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::split<float>(b, 0, 0x1p-20)),
                       doctest::Contains("The bound 9.5367431640625e-07 is within the tolerance"),
                       pantr::value_error);
  // Without the knots 0.5 and 0.75 the previous release refused with the knot-count error.
  const bs::Bspline<float> short_b =
      squares_curve<float>({-1, -1, -1, k3, 0x1p-20, 1, 1, 1}, 2, false);
  check_clamped_on(bs::restrict<float>(short_b, BoundsList{bs::Bounds{0x1p-20, 1.0}}), 0x1p-20,
                   1.0);
}

TEST_CASE("restrict and split: a NaN bound is refused, not cut as if every knot were at it") {
  // No knot equals a NaN and every comparison with one is false, so a search for the knots at it
  // has no meaning: it is refused before any count or cut.
  const bs::Bspline<double> b = squares_curve<double>({0, 0, 0, 0.5, 1, 1, 1}, 2, true);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::restrict<double>(b, BoundsList{bs::Bounds{nan, 0.7}})),
                       doctest::Contains("Lower bound (nan) is outside the domain"),
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::restrict<double>(b, BoundsList{bs::Bounds{0.2, nan}})),
                       doctest::Contains("Upper bound (nan) is outside the domain"),
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::split<double>(b, 0, nan)),
                       doctest::Contains("value must be strictly inside the domain"),
                       pantr::value_error);
}

// ---- the pieces of restrict and split keep the parent's tolerance
// -----------------------------------------------------------------
//
// The result's space used to be built with a tolerance derived from its own knots. Where the far
// knot `1e6` set the parent's, the window [0, 2] derived a smaller one, and `1` and `1 + 1e-12`,
// one knot of the unsnapped parent, became two knots of the result. `BsplineSpace1D::restrict`
// keeps the parent's tolerance, and so must these.

TEST_CASE("restrict and split: the pieces keep the parent direction's tolerance") {
  const bs::Bspline<double> b =
      squares_curve<double>({0, 0, 0, 1, 1 + 1e-12, 2, 3, 4, 1e6, 1e6, 1e6}, 2, false);
  const double tol = b.space().spaces()[0].tolerance();
  // 8 eps max(span, |k0|, |k_last|) = 8 eps 1e6, as the issue prints it.
  REQUIRE(tol == 1.7763568394002505e-09);
  const bs::Bspline<double> r = bs::restrict<double>(b, BoundsList{bs::Bounds{0.0, 2.0}});
  CHECK(r.space().spaces()[0].tolerance() == tol);
  CHECK(r.space().spaces()[0].num_intervals() == 2);
  const auto [left, right] = bs::split<double>(b, 0, 2.0);
  CHECK(left.space().spaces()[0].tolerance() == tol);
  CHECK(left.space().spaces()[0].num_intervals() == 2);
  CHECK(right.space().spaces()[0].tolerance() == tol);
}

TEST_CASE("restrict and split: a direction not cut keeps its space") {
  // Direction 1 has its own tolerance; the pieces cut along direction 0 keep it unchanged.
  const std::vector<double> k0 = {0, 0, 0, 1, 1 + 1e-12, 2, 3, 4, 1e6, 1e6, 1e6};
  const std::vector<double> k1 = {0, 0, 0.5, 3, 3};
  std::vector<bs::BsplineSpace1D<double>> spaces;
  spaces.emplace_back(sp(k0), 2, false, false);
  spaces.emplace_back(sp(k1), 1, false, false);
  const auto n = static_cast<std::size_t>(spaces[0].num_basis() * spaces[1].num_basis());
  const std::vector<double> cp(n, 1.0);
  const bs::Bspline<double> b(bs::BsplineSpace<double>(std::move(spaces)), sp(cp), false);
  const double tol0 = b.space().spaces()[0].tolerance();
  const double tol1 = b.space().spaces()[1].tolerance();
  const bs::Bspline<double> r =
      bs::restrict<double>(b, BoundsList{bs::Bounds{0.0, 2.0}, std::nullopt});
  CHECK(r.space().spaces()[0].tolerance() == tol0);
  CHECK(r.space().spaces()[1].tolerance() == tol1);
  const auto [left, right] = bs::split<double>(b, 0, 2.0);
  CHECK(left.space().spaces()[1].tolerance() == tol1);
  CHECK(right.space().spaces()[1].tolerance() == tol1);
}

TEST_CASE("restrict and split: the ends the review pinned, one by one") {
  const double eps = std::numeric_limits<double>::epsilon();
  // The upper end of a restriction refuses an unsnapped knot just below it, as the lower end does
  // one just above.
  const bs::Bspline<double> below =
      squares_curve<double>({0, 0, 0, 1.0, 2.0 - 32 * eps, 2.0, 4.0, 4.0, 4.0}, 2, false);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(bs::restrict<double>(below, BoundsList{bs::Bounds{0.5, 2.0}})),
      doctest::Contains("The bound 2.0 is within the tolerance"), pantr::value_error);
  check_clamped_on(bs::restrict<double>(below, BoundsList{bs::Bounds{2.0, 4.0}}), 2.0, 4.0);

  // An open domain end the cut keeps is the space's own, even with an unsnapped knot in its class.
  const bs::Bspline<double> start =
      squares_curve<double>({0, 0, 0, 8 * eps, 1.0, 2.0, 2.0, 2.0}, 2, false);
  REQUIRE(8 * eps <= start.space().spaces()[0].tolerance());
  const bs::Bspline<double> kept = bs::restrict<double>(start, BoundsList{bs::Bounds{0.0, 1.5}});
  const auto kk = kept.space().spaces()[0].knots();
  CHECK(kk[2] == 0.0);
  CHECK(kk[3] == 8 * eps);

  // Two bounds within the tolerance of each other enclose no interval.
  const bs::Bspline<double> b = squares_curve<double>({0, 0, 0, 0.5, 1, 1, 1}, 2, true);
  const double tol = b.space().spaces()[0].tolerance();
  CHECK_THROWS_WITH_AS(
      static_cast<void>(bs::restrict<double>(b, BoundsList{bs::Bounds{0.3, 0.3 + 0.5 * tol}})),
      doctest::Contains("enclose no interval"), pantr::value_error);
  check_clamped_on(bs::restrict<double>(b, BoundsList{bs::Bounds{0.3, 0.3 + 2 * tol}}), 0.3,
                   0.3 + 2 * tol);

  // A direct split_1d at NaN, which no knot equals, refuses rather than cutting at index 0.
  const std::vector<double> k = {0, 0, 0, 0.5, 1, 1, 1};
  const std::vector<double> cp = {1, 2, 3, 4};
  const mdview<const double, 2> ctrl(cp.data(), 4, 1);
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::split_1d<double>(
                           sp(k), 2, ctrl, false, tol, std::numeric_limits<double>::quiet_NaN())),
                       doctest::Contains("NaN"), pantr::value_error);
}

// ---- a periodic space whose seam knot is repeated -----------------------------------------------
//
// A periodic knot vector whose seam knot is repeated and whose domain starts at the first copy
// holds copies of the domain start `a = knots[p]` past index `p`: inside the domain, outside the
// first `p + 1` knots. A non-periodic vector can hold copies of an end inside its domain past an
// external knot too. Opening either counts every copy of a domain end.

namespace {

/// The periodic knot vector of period 1 whose domain knots are `cycle` and `1`:
/// `cycle.size() + 2 p + 1` knots, `p` ghost knots on each side of `[0, 1]`, `knots[p]` the first
/// copy of the seam `0`.
std::vector<double> periodic_knots(const std::vector<double>& cycle, int p) {
  const auto len = static_cast<long long>(cycle.size());
  const auto n = len + 2 * static_cast<long long>(p) + 1;
  std::vector<double> k;
  for (long long i = 0; i < n; ++i) {
    const long long j = i - p;
    const long long period = j >= 0 ? j / len : -((-j + len - 1) / len);
    k.push_back(cycle[static_cast<std::size_t>(j - period * len)] + static_cast<double>(period));
  }
  return k;
}

/// Check that `piece` is clamped on `[lo, hi]` and agrees with the periodic net `per` (of
/// `n_stored` stored points) there, at 19 interior points, within `tol`.
template <class T>
void check_agrees(const ExactNet& per, std::size_t n_stored, const bs::Bspline<T>& piece, double lo,
                  double hi, double tol) {
  CHECK_FALSE(piece.space().spaces()[0].periodic());
  check_clamped_on(piece, lo, hi);
  const auto ps = piece.space().spaces()[0];
  const std::vector<double> pk(ps.knots().begin(), ps.knots().end());
  const std::vector<double> pc(piece.control_points().begin(), piece.control_points().end());
  const ExactNet on = exact_net(pk, per.p, pc, 1);
  for (int j = 1; j < 20; ++j) {
    const Rational x = oracle::exact(lo + (hi - lo) * static_cast<double>(j) / 20.0);
    Rational d = exact_curve(per, x, n_stored)[0] - exact_curve(on, x)[0];
    if (d < 0) {
      d = -d;
    }
    CHECK(static_cast<double>(d) <= tol);
  }
}

}  // namespace

TEST_CASE_TEMPLATE("to_open, restrict and split: a periodic space with a repeated seam knot", T,
                   double, float) {
  struct SeamCase {
    std::vector<double> cycle;
    int p;
    std::size_t copies;
  };
  // With p + 1 copies (degree 1, a double seam) nothing is inserted and only the ghosts are
  // trimmed.
  for (const SeamCase& c :
       {SeamCase{{0.0, 0.0, 0.25, 0.5, 0.75}, 2, 2}, SeamCase{{0.0, 0.0, 0.25, 0.5, 0.75}, 3, 2},
        SeamCase{{0.0, 0.0, 0.0, 0.25, 0.5, 0.75}, 3, 3},
        SeamCase{{0.0, 0.0, 0.25, 0.5, 0.75}, 1, 2}}) {
    CAPTURE(c.p);
    CAPTURE(c.copies);
    const std::vector<double> knots = periodic_knots(c.cycle, c.p);
    const auto p = static_cast<std::size_t>(c.p);
    // The premise: copies of `a = 0` past index p.
    for (std::size_t i = 0; i < c.copies; ++i) {
      REQUIRE(knots[p + i] == 0.0);
    }
    REQUIRE(knots[p + c.copies] > 0.0);
    REQUIRE(knots[p - 1] < 0.0);
    const bs::BsplineSpace1D<T> ps = space_1d<T>(knots, c.p, true);
    // One function per knot of the cycle, as the oracle wraps the stored points.
    REQUIRE(static_cast<std::size_t>(ps.num_basis()) == c.cycle.size());
    const std::size_t n_stored = c.cycle.size();
    const std::vector<double> flat = values(n_stored, 11);
    const bs::Bspline<T> b = curve<T>(knots, c.p, flat, true);
    const ExactNet per = exact_net(knots, c.p, flat, 1);
    // One insertion pass rounds the control points by 8 (p + 1) u max|P| (the knot insertion
    // tests above), and max|P| <= 1. `restrict` and `split` insert once more on the opened
    // points: the exact refinement is a convex combination, so it carries the first pass's error
    // without growth and adds its own, two passes in all (to first order in u).
    const double pass = 8.0 * (static_cast<double>(c.p) + 1.0) * unit_roundoff<T>();
    check_agrees(per, n_stored, bs::to_open_bspline<T>(b), 0.0, 1.0, pass);
    const auto lo = static_cast<double>(static_cast<T>(0.3));
    const auto hi = static_cast<double>(static_cast<T>(0.6));
    check_agrees(per, n_stored, bs::restrict<T>(b, BoundsList{bs::Bounds{0.3, 0.6}}), lo, hi,
                 2.0 * pass);
    const auto [left, right] = bs::split<T>(b, 0, 0.5);
    check_agrees(per, n_stored, left, 0.0, 0.5, 2.0 * pass);
    check_agrees(per, n_stored, right, 0.5, 1.0, 2.0 * pass);
  }
}

TEST_CASE("to_open, restrict and split: an unsnapped knot within the tolerance of the seam") {
  // Each cycle holds a knot distinct from the seam and within the space's tolerance of it (the
  // knot vector spans 1.75, so the tolerance is about 3e-15): next to the start, or next to the
  // end.
  struct NearCase {
    std::vector<double> cycle;
    std::size_t near_index;
    double seam;
    const char* message;
  };
  for (const NearCase& c :
       {NearCase{{0.0, 1e-15, 0.25, 0.5, 0.75}, 3, 0.0, "The seam 0.0 is within the tolerance "},
        NearCase{{0.0, 0.25, 0.5, 0.75, 1.0 - 1e-15},
                 6,
                 1.0,
                 "The seam 1.0 is within the tolerance "}}) {
    CAPTURE(c.seam);
    const std::vector<double> knots = periodic_knots(c.cycle, 2);
    const bs::BsplineSpace1D<double> ps(sp(knots), 2, true, false);
    // The premise: the near knot is kept, distinct and within the tolerance.
    REQUIRE(ps.knots()[c.near_index] != c.seam);
    REQUIRE(std::abs(ps.knots()[c.near_index] - c.seam) <= ps.tolerance());
    const auto n_stored = static_cast<std::size_t>(ps.num_basis());
    const std::vector<double> cp = values(n_stored, 5);
    const bs::Bspline<double> b(bs::BsplineSpace<double>({ps}), sp(cp), false);
    const std::string by_seam = "distinct knot " + pantr::py_repr(ps.knots()[c.near_index]) + ",";
    const auto message_of = [&](auto&& call) -> std::string {
      try {
        call();
      } catch (const pantr::value_error& e) {
        return e.what();
      }
      return {};
    };
    for (const std::string& m :
         {message_of([&] { static_cast<void>(bs::to_open_bspline<double>(b)); }), message_of([&] {
            static_cast<void>(bs::restrict<double>(b, BoundsList{bs::Bounds{0.3, 0.6}}));
          }),
          message_of([&] { static_cast<void>(bs::split<double>(b, 0, 0.5)); })}) {
      CAPTURE(m);
      CHECK(m.starts_with(c.message));
      CHECK(m.find(by_seam) != std::string::npos);
      CHECK(m.ends_with("build the space with snap_knots=True."));
    }
  }
}

TEST_CASE("to_open_1d: copies of a domain end outside its first or last degree + 1 knots") {
  const auto open_of = [](const std::vector<double>& k, double tol) {
    static const std::vector<double> cp = {1.0, 2.0, 3.0, 4.0, 5.0};
    return bs::to_open_1d<double>(sp(k), 2, mdview<const double, 2>(cp.data(), k.size() - 3, 1),
                                  false, tol);
  };
  const auto rows = [](const bs::KnotsAndCtrl<double>& r) {
    return std::vector<double>(r.ctrl.data(), r.ctrl.data() + r.ctrl.size());
  };
  // Three copies of the start at indices 1 to 3 after an external knot, and three of the end at 4
  // to 6 before one: nothing is inserted, the external knot and its control point are trimmed.
  const bs::KnotsAndCtrl<double> start = open_of({-1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0}, 1e-12);
  CHECK(start.knots == std::vector<double>{0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0});
  CHECK(rows(start) == std::vector<double>{2.0, 3.0, 4.0, 5.0});
  const bs::KnotsAndCtrl<double> end = open_of({0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 3.0}, 1e-12);
  CHECK(end.knots == std::vector<double>{0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0});
  CHECK(rows(end) == std::vector<double>{1.0, 2.0, 3.0, 4.0});
  // An external knot within the tolerance below the start is not a copy of it: the result starts
  // at 0 (counted under the tolerance it started at -1e-15). The knots are those of inserting 0
  // once and 1.5 twice and trimming one knot left and two right.
  const bs::KnotsAndCtrl<double> near_start =
      open_of({-1e-15, 0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0}, 1e-12);
  CHECK(near_start.knots == std::vector<double>{0.0, 0.0, 0.0, 0.5, 1.0, 1.5, 1.5, 1.5});
  // A distinct knot within the tolerance of a non-periodic end, inside the domain, is refused
  // naming the domain end.
  CHECK_THROWS_WITH_AS(
      static_cast<void>(open_of({-1.0, 0.0, 0.0, 1e-15, 1.0, 2.0, 3.0, 4.0}, 1e-12)),
      doctest::Contains("The domain end 0.0 is within the tolerance 1e-12 of the "
                        "distinct knot 1e-15,"),
      pantr::value_error);
  // More than degree + 1 copies of a domain end is not a knot vector of that degree, at either end.
  const std::vector<double> cp = {1.0, 2.0, 3.0, 4.0, 5.0};
  for (const auto& [over, message] :
       {std::pair{std::vector<double>{0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0},
                  "to_open: the domain end 0.0 has 4 copies, more than degree + 1 = 3."},
        std::pair{std::vector<double>{0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0},
                  "to_open: the domain end 1.0 has 4 copies, more than degree + 1 = 3."}}) {
    CHECK_THROWS_WITH_AS(static_cast<void>(bs::to_open_1d<double>(
                             sp(over), 2, mdview<const double, 2>(cp.data(), 2, 1), true, 1e-12)),
                         message, pantr::value_error);
  }
}

TEST_CASE("to_open_1d: the end mirror of an external knot near a domain end, and the tolerance") {
  const std::vector<double> cp = {1.0, 2.0, 3.0, 4.0, 5.0};
  const auto open_of = [&](const std::vector<double>& k) {
    return bs::to_open_1d<double>(sp(k), 2, mdview<const double, 2>(cp.data(), 5, 1), false, 1e-12);
  };
  // An external knot within the tolerance above the end is not a copy of it: the result ends at
  // 1.5, the knots those of inserting 0 once and 1.5 twice and trimming one knot left and two
  // right.
  CHECK(open_of({-1.0, 0.0, 0.0, 0.5, 1.0, 1.5, 1.5 + 1e-15, 3.0}).knots ==
        std::vector<double>{0.0, 0.0, 0.0, 0.5, 1.0, 1.5, 1.5, 1.5});
  // A distinct knot inside the domain at exactly the tolerance from the start is refused, and one
  // an ulp further is kept.
  CHECK_THROWS_WITH_AS(static_cast<void>(open_of({-1.0, 0.0, 0.0, 1e-12, 1.0, 2.0, 2.0, 2.0})),
                       doctest::Contains("The domain end 0.0 is within the tolerance 1e-12"),
                       pantr::value_error);
  const double above = std::nextafter(1e-12, 1.0);
  CHECK(open_of({-1.0, 0.0, 0.0, above, 1.0, 2.0, 2.0, 2.0}).knots ==
        std::vector<double>{0.0, 0.0, 0.0, above, 1.0, 2.0, 2.0, 2.0});
}
