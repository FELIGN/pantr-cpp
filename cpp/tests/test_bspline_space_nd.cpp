// BsplineSpace, the tensor product of 1D spaces: its construction, the tensor-product basis at
// points and on a lattice (C order over the directions), the first function per direction, the
// refusals and their text, and the pool. Oracles: products of the exact-rational 1D bases
// (bspline_oracle.hpp) assembled in C order by a loop of its own, closed-form Bernstein products
// worked out by hand for the index order, and the agreement of the points and lattice overloads
// (two code paths to the same rows).
#include <doctest/doctest.h>

#include "bspline_oracle.hpp"

#include <pantr/bspline/knot_factories.hpp>
#include <pantr/bspline/space.hpp>
#include <pantr/parallel/parallel_for.hpp>
#include <pantr/parallel/work_model.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace bs = pantr::bspline;
namespace oracle = bspline_oracle;
using pantr::mdview;
using pantr::NdArray;

namespace {

template <class T>
double eps_of() {
  return static_cast<double>(std::numeric_limits<T>::epsilon());
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

/// The value of `v` as the type `P` holds it, as a double.
template <class P>
double as_p(double v) {
  return static_cast<double>(static_cast<P>(v));
}

/// A storage type of the space and a point type.
template <class K, class P>
struct KP {
  using Knot = K;
  using Point = P;
};
using DD = KP<double, double>;
using FF = KP<float, float>;
using FD = KP<float, double>;
using DF = KP<double, float>;

/// One direction of a test space: dyadic knots, so the same space exists in float and double.
struct Dir {
  std::vector<double> knots;
  int degree;
};

const Dir kQuad = {{0, 0, 0, 0.25, 0.75, 0.75, 1, 1, 1}, 2};
const Dir kHat = {{0, 0, 0.5, 1, 1}, 1};
const Dir kCubic = {{0, 0, 0, 0, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4}, 3};
const Dir kBez2 = {{0, 0, 0, 2, 2, 2}, 2};  // Bezier on [0, 2]
const Dir kBez1 = {{0, 0, 1, 1}, 1};        // Bezier on [0, 1]

template <class T>
bs::BsplineSpace<T> make_space(const std::vector<Dir>& dirs) {
  std::vector<bs::BsplineSpace1D<T>> spaces;
  for (const Dir& d : dirs) {
    const std::vector<T> k = cast_all<T>(d.knots);
    spaces.emplace_back(sp(k), d.degree);
  }
  return bs::BsplineSpace<T>(std::move(spaces));
}

/// The coordinates a direction is sampled at: its knots, the midpoints between them and a point
/// at 0.8 of each interval (not dyadic).
std::vector<double> coordinates(const Dir& d) {
  const auto p = static_cast<std::size_t>(d.degree);
  std::vector<double> ks;
  for (std::size_t i = p; i + p + 1 <= d.knots.size(); ++i) {
    if (ks.empty() || d.knots[i] != ks.back()) {
      ks.push_back(d.knots[i]);
    }
  }
  std::vector<double> out = ks;
  for (std::size_t i = 0; i + 1 < ks.size(); ++i) {
    out.push_back(0.5 * (ks[i] + ks[i + 1]));
    out.push_back(ks[i] + 0.8 * (ks[i + 1] - ks[i]));
  }
  return out;
}

/// The exact rows of the tensor product at one point: for each direction the exact 1D basis, then
/// the products assembled with the last direction varying fastest (a list grown direction by
/// direction, not the in-place expansion the library uses).
struct ExactRow {
  std::vector<std::int64_t> first;
  std::vector<oracle::Rational> values;
};

template <class T>
ExactRow exact_row(const bs::BsplineSpace<T>& space, const std::vector<double>& x) {
  ExactRow row;
  row.values = {oracle::Rational(1)};
  for (std::size_t d = 0; d < space.dim(); ++d) {
    const auto& s = space.spaces()[d];
    const int p = s.degree();
    oracle::BasisAt b(oracle::exact_knots<T>(s.knots()), p, oracle::exact(x[d]));
    const auto f = static_cast<std::int64_t>(b.span()) - p;
    row.first.push_back(f);
    std::vector<oracle::Rational> grown;
    for (const oracle::Rational& v : row.values) {
      for (int m = 0; m < p + 1; ++m) {
        grown.push_back(v * b.deriv(static_cast<std::size_t>(f) + static_cast<std::size_t>(m), 0));
      }
    }
    row.values = std::move(grown);
  }
  return row;
}

/// The relative bound on one entry: each 1D factor within `8 p_d eps` (see test_bspline_space.cpp),
/// the `D - 1` products and the final store one rounding each, so `(8 sum p + D + 1) eps`.
template <class T>
double product_bound(const bs::BsplineSpace<T>& space, double eps) {
  double sum_p = 0.0;
  for (const auto& s : space.spaces()) {
    sum_p += s.degree();
  }
  return (8.0 * sum_p + static_cast<double>(space.dim()) + 1.0) * eps;
}

/// Compare a table of rows (`out`, `first`) with the exact products at the points `xs`.
template <class T, class P, class O>
void check_rows(const bs::BsplineSpace<T>& space, const std::vector<std::vector<double>>& xs,
                const NdArray<O>& out, const NdArray<std::int64_t>& first) {
  const std::size_t nfun = space.num_local_functions();
  const double eps = (std::max)({eps_of<T>(), eps_of<P>(), eps_of<O>()});
  const double bound = product_bound(space, eps);
  REQUIRE(out.shape()[0] == xs.size());
  REQUIRE(out.shape()[1] == nfun);
  for (std::size_t i = 0; i < xs.size(); ++i) {
    const ExactRow row = exact_row(space, xs[i]);
    REQUIRE(row.values.size() == nfun);
    CAPTURE(i);
    for (std::size_t d = 0; d < space.dim(); ++d) {
      CAPTURE(d);
      CHECK(first.data()[i * space.dim() + d] == row.first[d]);
    }
    for (std::size_t j = 0; j < nfun; ++j) {
      CAPTURE(j);
      const double want = row.values[j].template convert_to<double>();
      CHECK(oracle::within(static_cast<double>(out.data()[i * nfun + j]), row.values[j],
                           bound * std::fabs(want)));
    }
  }
}

}  // namespace

// ---- construction -----------------------------------------------------------------------------

TEST_CASE("space nd: a space needs at least one direction") {
  CHECK_THROWS_WITH_AS(
      static_cast<void>(bs::BsplineSpace<double>(std::vector<bs::BsplineSpace1D<double>>{})),
      "a B-spline space needs at least one direction", pantr::value_error);
}

TEST_CASE("space nd: counts that overflow are a capacity_error, as many as fit are not") {
  // A Bezier direction of degree 70000 has 70001 functions and local functions: three directions
  // multiply to 3.4e14, four to 2.4e19, past the 1.8e19 of std::size_t. Left unchecked, the count
  // wraps and the output buffers a caller sizes from it are too small.
  const std::vector<double> knots = bs::uniform_open_knots<double>(1, 70000, 69999, 0.0, 1.0);
  const bs::BsplineSpace1D<double> s(sp(knots), 70000);
  const std::vector<bs::BsplineSpace1D<double>> three(3, s);
  const std::vector<bs::BsplineSpace1D<double>> four(4, s);
  CHECK_NOTHROW(static_cast<void>(bs::BsplineSpace<double>(three)));
  CHECK_THROWS_AS(static_cast<void>(bs::BsplineSpace<double>(four)), pantr::capacity_error);
  // Only the count of basis functions overflows here: 20000 cubic intervals discontinuous at every
  // knot have 80000 functions per direction (4.1e19 for four), against 20000 intervals (1.6e17)
  // and 4 local functions (256).
  const std::vector<double> many = bs::uniform_open_knots<double>(20000, 3, -1, 0.0, 1.0);
  const bs::BsplineSpace1D<double> wide(sp(many), 3);
  REQUIRE(wide.num_basis() == 80000);
  const std::vector<bs::BsplineSpace1D<double>> wide_four(4, wide);
  CHECK_THROWS_AS(static_cast<void>(bs::BsplineSpace<double>(wide_four)), pantr::capacity_error);
}

TEST_CASE("space nd: the shape of the product and its derived properties") {
  const auto space = make_space<double>({kQuad, kHat, kCubic});
  CHECK(space.dim() == 3);
  CHECK(space.num_local_functions() == 3 * 2 * 4);
  CHECK_FALSE(space.has_bezier_like_knots());
  CHECK(make_space<double>({kBez1, kBez2}).has_bezier_like_knots());
  CHECK_FALSE(make_space<double>({kBez1, kHat}).has_bezier_like_knots());
  // The tolerance is the largest of the directions': 8 eps scale of [0, 4] against [0, 1].
  CHECK(space.tolerance() == space.spaces()[2].tolerance());
  CHECK(space.tolerance() > space.spaces()[0].tolerance());
  CHECK(space.tolerance() == 8.0 * eps_of<double>() * 4.0);
}

// ---- the tensor product -----------------------------------------------------------------------

TEST_CASE("space nd: the index order of the product is C order, worked out by hand") {
  // Degrees (1, 2, 3) on [0, 1]^3 as Bezier pieces: the entry (a, b, c) of the 24 is
  // L_a(x) B2_b(y) B3_c(z) at flat position (a * 3 + b) * 4 + c. Linear, quadratic and cubic
  // Bernstein values at 0.25, 0.5, 0.75, written out from the binomial formula.
  const Dir lin = {{0, 0, 1, 1}, 1};
  const Dir quad = {{0, 0, 0, 1, 1, 1}, 2};
  const Dir cub = {{0, 0, 0, 0, 1, 1, 1, 1}, 3};
  const auto space = make_space<double>({lin, quad, cub});
  const double l[2] = {0.75, 0.25};
  const double q[3] = {0.25, 0.5, 0.25};
  const double c[4] = {0.015625, 0.140625, 0.421875, 0.421875};
  const std::vector<double> pts = {0.25, 0.5, 0.75};
  NdArray<double> out({1, 24});
  NdArray<std::int64_t> first({1, 3});
  space.tabulate_basis<double>(mdview<const double, 2>(pts.data(), 1, 3), out.view<2>(),
                               first.view<2>());
  for (std::size_t a = 0; a < 2; ++a) {
    for (std::size_t b = 0; b < 3; ++b) {
      for (std::size_t k = 0; k < 4; ++k) {
        CAPTURE(a);
        CAPTURE(b);
        CAPTURE(k);
        CHECK(std::fabs(out.data()[(a * 3 + b) * 4 + k] - l[a] * q[b] * c[k]) <=
              16.0 * eps_of<double>());
      }
    }
  }
  CHECK(first.data()[0] == 0);
  CHECK(first.data()[1] == 0);
  CHECK(first.data()[2] == 0);
}

TEST_CASE_TEMPLATE("space nd: the rows at points are the products of the exact 1D bases", Pair, DD,
                   FF, FD, DF) {
  using T = typename Pair::Knot;
  using P = typename Pair::Point;
  const auto space = make_space<T>({kQuad, kHat, kCubic});
  const std::vector<std::vector<double>> axes = {coordinates(kQuad), coordinates(kHat),
                                                 coordinates(kCubic)};
  std::vector<std::vector<double>> xs;
  std::vector<P> flat;
  for (std::size_t i = 0; i < 40; ++i) {
    // Each direction at a different stride, so the combinations differ from row to row.
    const double x = as_p<P>(axes[0][i % axes[0].size()]);
    const double y = as_p<P>(axes[1][(3 * i + 1) % axes[1].size()]);
    const double z = as_p<P>(axes[2][(7 * i + 2) % axes[2].size()]);
    xs.push_back({x, y, z});
    flat.insert(flat.end(), {static_cast<P>(x), static_cast<P>(y), static_cast<P>(z)});
  }
  const auto [out, first] =
      space.template tabulate_basis<P>(mdview<const P, 2>(flat.data(), xs.size(), 3));
  check_rows<T, P, T>(space, xs, out, first);
}

TEST_CASE_TEMPLATE("space nd: a lattice is tabulated in C order, row p for the p-th point", Pair,
                   DD, FF, FD, DF) {
  using T = typename Pair::Knot;
  using L = typename Pair::Point;
  const auto space = make_space<T>({kQuad, kCubic});
  const std::vector<double> x0 = {0.0, 0.25, 0.5, 1.0};
  const std::vector<double> x1 = {0.0, 1.0, 1.5, 2.0, 3.5};
  const pantr::quad::PointsLattice<L> lattice({cast_all<L>(x0), cast_all<L>(x1)});
  NdArray<L> out({lattice.num_points(), space.num_local_functions()}, pantr::for_overwrite);
  NdArray<std::int64_t> first({lattice.num_points(), 2}, pantr::for_overwrite);
  space.template tabulate_basis<L>(lattice, out.template view<2>(), first.template view<2>());
  // The points of the lattice, the last direction fastest: (x0[i], x1[j]) at row i * 5 + j.
  std::vector<std::vector<double>> xs;
  for (const double a : x0) {
    for (const double b : x1) {
      xs.push_back({as_p<L>(a), as_p<L>(b)});
    }
  }
  check_rows<T, L, L>(space, xs, out, first);
}

TEST_CASE("space nd: a lattice and the same points as rows give the same table") {
  // Two code paths to one table: the lattice shares the 1D rows across points and decomposes the
  // start of each pool chunk into a multi-index, the points path tabulates each row on its own.
  // 100 x 80 = 8000 points of nine functions pass the serial threshold; the chunks are 3641 rows,
  // so the second chunk starts at (45, 41) of the lattice. Different counts per direction make
  // the C and F orders differ.
  std::vector<double> x0(100);
  std::vector<double> x1(80);
  for (std::size_t i = 0; i < x0.size(); ++i) {
    x0[i] = static_cast<double>(i) / 99.0;
  }
  for (std::size_t j = 0; j < x1.size(); ++j) {
    x1[j] = 2.0 * static_cast<double>(j) / 79.0;
  }
  const auto space = make_space<double>({kQuad, kBez2});
  const pantr::quad::PointsLattice<double> lattice({x0, x1});
  const std::size_t n = lattice.num_points();
  REQUIRE(n == 8000);
  REQUIRE(pantr::parallel::worth_parallel(
      n, pantr::parallel::WorkEstimate{2.0 * static_cast<double>(space.num_local_functions())}));
  NdArray<double> lat_out({n, space.num_local_functions()}, pantr::for_overwrite);
  NdArray<std::int64_t> lat_first({n, 2}, pantr::for_overwrite);
  space.tabulate_basis<double>(lattice, lat_out.view<2>(), lat_first.view<2>());
  const NdArray<double> all = lattice.all_points(pantr::Order::C);
  const auto [pt_out, pt_first] = space.tabulate_basis<double>(all.view<2>());
  REQUIRE(lat_out.size() == pt_out.size());
  CHECK(std::equal(lat_first.data(), lat_first.data() + lat_first.size(), pt_first.data()));
  double worst = 0.0;
  for (std::size_t i = 0; i < lat_out.size(); ++i) {
    worst = (std::max)(worst, std::fabs(lat_out.data()[i] - pt_out.data()[i]));
  }
  CHECK(worst <= 16.0 * eps_of<double>());
  // A spot check against the exact product at the rows around the chunk boundaries.
  for (const std::size_t r :
       {std::size_t{0}, std::size_t{3640}, std::size_t{3641}, std::size_t{3642}, std::size_t{7281},
        std::size_t{7282}, std::size_t{7999}}) {
    const ExactRow row = exact_row(space, {x0[r / 80], x1[r % 80]});
    CAPTURE(r);
    CHECK(lat_first.data()[r * 2] == row.first[0]);
    CHECK(lat_first.data()[r * 2 + 1] == row.first[1]);
    for (std::size_t j = 0; j < 9; ++j) {
      CHECK(std::fabs(lat_out.data()[r * 9 + j] - row.values[j].convert_to<double>()) <=
            16.0 * eps_of<double>());
    }
  }
}

TEST_CASE("space nd: one thread and the whole pool tabulate the same table") {
  // Chunks are a function of the problem only; each row is computed alone. The first indices are
  // equal exactly and the values within the rounding bound of two evaluations of the rows (each
  // within (8 sum p + 3) eps of the exact one, and no value exceeds 1).
  std::vector<double> x0(100);
  std::vector<double> x1(80);
  for (std::size_t i = 0; i < x0.size(); ++i) {
    x0[i] = static_cast<double>(i) / 99.0;
  }
  for (std::size_t j = 0; j < x1.size(); ++j) {
    x1[j] = 2.0 * static_cast<double>(j) / 79.0;
  }
  const auto space = make_space<double>({kQuad, kBez2});
  const pantr::quad::PointsLattice<double> lattice({x0, x1});
  const auto run = [&](std::size_t threads) {
    const std::size_t previous = pantr::parallel::get_num_threads();
    pantr::parallel::set_num_threads(static_cast<std::int64_t>(threads));
    NdArray<double> o({lattice.num_points(), space.num_local_functions()}, pantr::for_overwrite);
    NdArray<std::int64_t> f({lattice.num_points(), 2}, pantr::for_overwrite);
    space.tabulate_basis<double>(lattice, o.view<2>(), f.view<2>());
    pantr::parallel::set_num_threads(static_cast<std::int64_t>(previous));
    return std::make_pair(std::move(o), std::move(f));
  };
  const auto serial = run(1);
  WARN_MESSAGE(pantr::parallel::max_threads() >= 2, "one CPU: the pooled run is the serial one");
  const auto pooled = run(pantr::parallel::max_threads());
  CHECK(std::equal(serial.second.data(), serial.second.data() + serial.second.size(),
                   pooled.second.data()));
  double worst = 0.0;
  for (std::size_t i = 0; i < serial.first.size(); ++i) {
    worst = (std::max)(worst, std::fabs(serial.first.data()[i] - pooled.first.data()[i]));
  }
  CHECK(worst <= 2.0 * (8.0 * 4.0 + 3.0) * eps_of<double>());
}

// ---- the first direction outside --------------------------------------------------------------

TEST_CASE_TEMPLATE("space nd: the first direction with a point outside, or -1", P, float, double) {
  const auto space = make_space<double>({kBez1, kBez2});  // [0, 1] x [0, 2]
  const auto outside = [&](const std::vector<P>& flat) {
    return space.template first_direction_outside<P>(
        mdview<const P, 2>(flat.data(), flat.size() / 2, 2));
  };
  CHECK(outside({P(0.5), P(1.0), P(1.0), P(2.0)}) == -1);
  CHECK(outside({P(0.5), P(1.0), P(0.5), P(2.5)}) == 1);
  // Offenders in both directions, in different rows: the first direction in order wins, although
  // the offender in direction 1 comes in an earlier row.
  CHECK(outside({P(0.5), P(3.0), P(1.5), P(1.0)}) == 0);
  CHECK(outside({std::numeric_limits<P>::quiet_NaN(), P(1.0)}) == 0);
  const std::vector<P> three = {P(0.5), P(1.0), P(1.0)};
  CHECK_THROWS_WITH_AS(static_cast<void>(space.template first_direction_outside<P>(
                           mdview<const P, 2>(three.data(), 1, 3))),
                       "pts must have 2 columns", pantr::value_error);
  const pantr::quad::PointsLattice<P> inside({std::vector<P>{P(0.5)}, std::vector<P>{P(2.0)}});
  const pantr::quad::PointsLattice<P> out_in_1({std::vector<P>{P(0.5)}, std::vector<P>{P(2.5)}});
  const pantr::quad::PointsLattice<P> out_in_0(
      {std::vector<P>{P(0.5), P(1.5)}, std::vector<P>{P(2.5)}});
  const pantr::quad::PointsLattice<P> one_dim({std::vector<P>{P(0.5)}});
  CHECK(space.template first_direction_outside<P>(inside) == -1);
  CHECK(space.template first_direction_outside<P>(out_in_1) == 1);
  CHECK(space.template first_direction_outside<P>(out_in_0) == 0);
  CHECK_THROWS_WITH_AS(static_cast<void>(space.template first_direction_outside<P>(one_dim)),
                       "pts must have 2 columns", pantr::value_error);
}

// ---- refusals ---------------------------------------------------------------------------------

TEST_CASE("space nd: refusals name the column or direction, with the matched text") {
  const auto space = make_space<double>({kBez1, kBez2});
  const std::vector<double> ok = {0.5, 1.0, 0.25, 2.0};
  const std::vector<double> bad1 = {0.5, 1.0, 0.25, 2.5};
  const std::vector<double> bad_both = {1.5, 3.0, 0.25, 1.0};
  NdArray<double> out({2, 6});
  NdArray<std::int64_t> first({2, 2});
  const auto call = [&](const std::vector<double>& pts) {
    space.tabulate_basis<double>(mdview<const double, 2>(pts.data(), 2, 2), out.view<2>(),
                                 first.view<2>());
  };
  CHECK_NOTHROW(call(ok));
  std::fill(out.data(), out.data() + 12, 7.0);
  CHECK_THROWS_WITH_AS(
      call(bad1),
      "One or more values in pts[:, 1] are outside the knot vector domain [[0.0, 1.0], [0.0, 2.0]]",
      pantr::value_error);
  // The first direction that has an offender is the one named.
  CHECK_THROWS_WITH_AS(
      call(bad_both),
      "One or more values in pts[:, 0] are outside the knot vector domain [[0.0, 1.0], [0.0, 2.0]]",
      pantr::value_error);
  // Nothing is written by a refused call.
  CHECK(std::all_of(out.data(), out.data() + 12, [](double x) { return x == 7.0; }));
  // Extents, and the column count of the convenience form.
  NdArray<double> narrow({2, 5});
  CHECK_THROWS_WITH_AS(space.tabulate_basis<double>(mdview<const double, 2>(ok.data(), 2, 2),
                                                    narrow.view<2>(), first.view<2>()),
                       "Output array has shape (2, 5), but expected shape (2, 6)",
                       pantr::value_error);
  NdArray<std::int64_t> wide({2, 3});
  CHECK_THROWS_WITH_AS(space.tabulate_basis<double>(mdview<const double, 2>(ok.data(), 2, 2),
                                                    out.view<2>(), wide.view<2>()),
                       "Output array has shape (2, 3), but expected shape (2, 2)",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(space.tabulate_basis<double>(mdview<const double, 2>(ok.data(), 1, 4))),
      "pts must have 2 columns", pantr::value_error);
  // A lattice: the direction is named as the Python wrapper's attribute.
  const pantr::quad::PointsLattice<double> lat_bad(
      {std::vector<double>{0.5}, std::vector<double>{2.5, 1.0}});
  NdArray<double> lout({2, 6});
  NdArray<std::int64_t> lfirst({2, 2});
  CHECK_THROWS_WITH_AS(
      space.tabulate_basis<double>(lat_bad, lout.view<2>(), lfirst.view<2>()),
      "One or more values in pts.pts_per_dir[1] are outside the knot vector domain (0.0, 2.0)",
      pantr::value_error);
  const pantr::quad::PointsLattice<double> lat_ok(
      {std::vector<double>{0.5, 0.25}, std::vector<double>{1.0}});
  NdArray<double> wrong_rows({3, 6});
  CHECK_THROWS_WITH_AS(space.tabulate_basis<double>(lat_ok, wrong_rows.view<2>(), lfirst.view<2>()),
                       "Output array has shape (3, 6), but expected shape (2, 6)",
                       pantr::value_error);
  const pantr::quad::PointsLattice<double> lat_1d({std::vector<double>{0.5}});
  CHECK_THROWS_WITH_AS(space.tabulate_basis<double>(lat_1d, lout.view<2>(), lfirst.view<2>()),
                       "pts must have 2 columns", pantr::value_error);
}

TEST_CASE_TEMPLATE("space nd: no points give an empty table", T, float, double) {
  const auto space = make_space<T>({kBez1, kQuad});
  const std::vector<T> none;
  const auto [out, first] = space.template tabulate_basis<T>(mdview<const T, 2>(none.data(), 0, 2));
  CHECK(out.size() == 0);
  CHECK(out.shape()[1] == 6);
  CHECK(first.size() == 0);
  CHECK(first.shape()[1] == 2);
}
