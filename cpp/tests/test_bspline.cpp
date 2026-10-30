// Bspline, the value (a space and control points) and its evaluation and partial derivatives.
//
// Oracles: the exact-rational Cox-de Boor definition of the basis (bspline_oracle.hpp), summed over
// the control points in exact arithmetic with the periodic index wrapped modulo the number of
// basis functions, and for a rational B-spline the quotient rule applied to the exact homogeneous
// derivatives; a rational quarter circle whose points satisfy x^2 + y^2 = 1 and whose derivatives
// satisfy the identities that follow from it, worked out by hand; and the agreement of the points,
// lattice and `out` overloads. Tolerances are derived in `poly_bound` and `exact_derivative` from
// the unit roundoff of the storage type.
#include <doctest/doctest.h>

#include "bspline_oracle.hpp"

#include <pantr/bspline/bspline.hpp>
#include <pantr/bspline/evaluate.hpp>
#include <pantr/bspline/knot_factories.hpp>
#include <pantr/parallel/pool.hpp>
#include <pantr/quad/points_lattice.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace bs = pantr::bspline;
namespace oracle = bspline_oracle;
using oracle::Rational;
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

template <class K, class P>
struct KP {
  using Knot = K;
  using Point = P;
};
using DD = KP<double, double>;
using FF = KP<float, float>;

/// One direction of a test space: dyadic knots, so the same space exists in float and double.
struct Dir {
  std::vector<double> knots;
  int degree;
  bool periodic = false;
};

const Dir kQuad = {{0, 0, 0, 0.25, 0.75, 0.75, 1, 1, 1}, 2};
const Dir kHat = {{0, 0, 0.5, 1, 1}, 1};
const Dir kCubic = {{0, 0, 0, 0, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4}, 3};
const Dir kNonOpen = {{-1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5}, 2};
// Domain [0, 2] with an interior knot of multiplicity two: C^0 there.
const Dir kQuadC0 = {{0, 0, 0, 1, 1, 2, 2, 2}, 2};

/// Four intervals of length 1/4 on [0, 1], periodic.
Dir periodic_dir(int continuity) {
  return {bs::uniform_periodic_knots<double>(4, 2, continuity, 0.0, 1.0), 2, true};
}

template <class T>
bs::BsplineSpace<T> make_space(const std::vector<Dir>& dirs) {
  std::vector<bs::BsplineSpace1D<T>> spaces;
  for (const Dir& d : dirs) {
    const std::vector<T> k = cast_all<T>(d.knots);
    spaces.emplace_back(sp(k), d.degree, d.periodic);
  }
  return bs::BsplineSpace<T>(std::move(spaces));
}

std::size_t total_basis(const std::vector<Dir>& dirs) {
  const auto space = make_space<double>(dirs);
  std::size_t n = 1;
  for (const auto& s : space.spaces()) {
    n *= static_cast<std::size_t>(s.num_basis());
  }
  return n;
}

/// Deterministic control points in [-1, 1], exactly representable in float (multiples of 1/64).
std::vector<double> control_values(std::size_t count, unsigned seed) {
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

/// A B-spline with `cp_size` coordinates per control point (the last a weight in [0.5, 2] when
/// rational).
template <class T>
bs::Bspline<T> make_bspline(const std::vector<Dir>& dirs, std::size_t cp_size, bool rational,
                            unsigned seed = 1) {
  const std::size_t n = total_basis(dirs);
  std::vector<double> v = control_values(n * cp_size, seed);
  if (rational) {
    for (std::size_t i = 0; i < n; ++i) {
      v[i * cp_size + cp_size - 1] = 0.5 + 1.5 * (v[i * cp_size + cp_size - 1] + 1.0) / 2.0;
    }
  }
  const std::vector<T> cp = cast_all<T>(v);
  return bs::Bspline<T>(make_space<T>(dirs), sp(cp), rational);
}

/// The coordinates a direction is sampled at: the ends of its domain, the midpoints and a point at
/// 0.8 of each interval (not dyadic).
template <class T>
std::vector<double> coordinates(const bs::BsplineSpace1D<T>& s) {
  const auto ks = s.unique_knots_and_multiplicity(true).first;
  std::vector<double> out;
  for (const T k : ks) {
    out.push_back(static_cast<double>(k));
  }
  for (std::size_t i = 0; i + 1 < ks.size(); ++i) {
    const auto a = static_cast<double>(ks[i]);
    const auto b = static_cast<double>(ks[i + 1]);
    out.push_back(0.5 * (a + b));
    out.push_back(a + 0.8 * (b - a));
  }
  return out;
}

/// The least length of an interval of a direction.
template <class T>
double min_interval(const bs::BsplineSpace1D<T>& s) {
  const auto ks = s.unique_knots_and_multiplicity(true).first;
  double h = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i + 1 < ks.size(); ++i) {
    h = (std::min)(h, static_cast<double>(ks[i + 1]) - static_cast<double>(ks[i]));
  }
  return h;
}

// ---- the exact oracle -------------------------------------------------------------------------

/// The exact homogeneous derivative of multi-index `ks` of the B-spline's sum over its control
/// points (the weight column included) at the point `x`.
template <class T>
std::vector<Rational> exact_hom(const bs::Bspline<T>& b, const std::vector<double>& x,
                                const std::vector<int>& ks) {
  const auto space = b.space();
  const auto& spaces = space.spaces();
  const std::size_t dim = b.dim();
  std::vector<oracle::BasisAt> ats;
  std::vector<std::size_t> first;
  std::vector<std::size_t> ord;
  for (std::size_t d = 0; d < dim; ++d) {
    const int p = spaces[d].degree();
    ats.emplace_back(oracle::exact_knots<T>(spaces[d].knots()), p, oracle::exact(x[d]));
    first.push_back(ats[d].span() - static_cast<std::size_t>(p));
    ord.push_back(static_cast<std::size_t>(p) + 1);
  }
  const auto cp = b.control_points();
  const std::size_t cps = b.cp_size();
  std::vector<Rational> out(cps, Rational(0));
  std::vector<std::size_t> j(dim, 0);
  for (;;) {
    Rational w(1);
    std::size_t flat = 0;
    for (std::size_t d = 0; d < dim; ++d) {
      w *= ats[d].deriv(first[d] + j[d], ks[d]);
      const std::size_t i = first[d] + j[d];
      const std::size_t nb = b.num_basis(d);
      flat = flat * nb + (spaces[d].periodic() ? i % nb : i);
    }
    if (w != 0) {
      for (std::size_t c = 0; c < cps; ++c) {
        out[c] += w * oracle::exact(static_cast<double>(cp[flat * cps + c]));
      }
    }
    std::size_t d = dim;
    for (;;) {
      if (d == 0) {
        return out;
      }
      --d;
      if (++j[d] < ord[d]) {
        break;
      }
      j[d] = 0;
    }
  }
}

double binom(int n, int k) {
  double c = 1.0;
  for (int i = 1; i <= k; ++i) {
    c = c * (n - k + i) / i;
  }
  return c;
}

/// The multi-indices `0 <= i <= ks`, by ascending total order (and then lexicographically).
std::vector<std::vector<int>> box(const std::vector<int>& ks) {
  std::vector<std::vector<int>> all = {{}};
  for (const int k : ks) {
    std::vector<std::vector<int>> grown;
    for (const auto& m : all) {
      for (int v = 0; v <= k; ++v) {
        auto g = m;
        g.push_back(v);
        grown.push_back(std::move(g));
      }
    }
    all = std::move(grown);
  }
  const auto total = [](const std::vector<int>& m) {
    int s = 0;
    for (const int v : m) {
      s += v;
    }
    return s;
  };
  std::stable_sort(
      all.begin(), all.end(),
      [&](const std::vector<int>& a, const std::vector<int>& b) { return total(a) < total(b); });
  return all;
}

bool leq(const std::vector<int>& i, const std::vector<int>& k) {
  for (std::size_t d = 0; d < i.size(); ++d) {
    if (i[d] > k[d]) {
      return false;
    }
  }
  return true;
}

/// An exact derivative of the projected map, and a bound on the error of the computed one.
struct Exact {
  std::vector<Rational> value;  // rank values
  std::vector<double> bound;    // rank bounds
};

/// First-order bound on the error of the polynomial sum of multi-index `ks`, for one coordinate.
///
/// One direction's derivative row of order `k_d` has entries at most `mag_d = (2 p_d / h_d)^k_d`
/// (each differentiation divides by a knot difference of at least `h_d` and combines two terms
/// of weight at most `p_d`), each within `16 (p_d + 1) eps mag_d` (the bound of the 1D tests, an
/// admitted margin over the roundings of the recurrence; for `k = 0` it contains `8 p eps`). The
/// weight of a local function is the product of `dim` such entries: its error is the sum over the
/// directions of one error times the other magnitudes, plus `dim` roundings (the products and the
/// store) of the product of the magnitudes. The sum over the `nloc` local functions adds `nloc`
/// roundings relative to the sum of the magnitudes `W = prod((p_d + 1) mag_d)`. With `cmax` the
/// largest control-point magnitude of the coordinate: `cmax (nloc dw + nloc eps W)`.
template <class T>
double poly_bound(const bs::Bspline<T>& b, const std::vector<int>& ks, double cmax, double eps) {
  const auto space = b.space();
  const auto& spaces = space.spaces();
  const std::size_t dim = b.dim();
  std::vector<double> mag(dim);
  std::vector<double> tol(dim);
  double nloc = 1.0;
  double wsum = 1.0;
  double prod_mag = 1.0;
  for (std::size_t d = 0; d < dim; ++d) {
    const double p = spaces[d].degree();
    const double h = min_interval(spaces[d]);
    mag[d] = std::pow(2.0 * (std::max)(p, 1.0) / h, ks[d]);
    tol[d] = 16.0 * (p + 1.0) * eps * mag[d];
    nloc *= p + 1.0;
    wsum *= (p + 1.0) * mag[d];
    prod_mag *= mag[d];
  }
  double dw = static_cast<double>(dim) * eps * prod_mag;
  for (std::size_t d = 0; d < dim; ++d) {
    double others = 1.0;
    for (std::size_t e = 0; e < dim; ++e) {
      others *= e == d ? 1.0 : mag[e];
    }
    dw += tol[d] * others;
  }
  return cmax * (nloc * dw + nloc * eps * wsum);
}

/// The largest magnitude of each coordinate (column) of the control points.
template <class T>
std::vector<double> column_max(const bs::Bspline<T>& b) {
  std::vector<double> m(b.cp_size(), 0.0);
  const auto cp = b.control_points();
  for (std::size_t i = 0; i < cp.size(); ++i) {
    m[i % b.cp_size()] = (std::max)(m[i % b.cp_size()], std::fabs(static_cast<double>(cp[i])));
  }
  return m;
}

/// The exact derivative of multi-index `ks` of the (projected) map at `x`, and a bound on the
/// error of the computed one.
///
/// A rational map is `R_k = (H_k - sum_{0 < i <= k} C(k, i) w_i R_{k - i}) / w_0` over the
/// homogeneous derivatives `H`, formed in exact arithmetic here. To first order, with `E_k` the
/// polynomial bound on the error of `H_k`, `dR_k = (dH_k - sum C (dw_i R_{k-i} + w_i dR_{k-i}) -
/// R_k dw_0) / w_0`, plus the roundings of the sum and the division, `(terms + 2) eps` of the sum
/// of magnitudes. The second-order terms are of the order of `eps` times the first and are covered
/// by the factor 2 of the result.
template <class T>
Exact exact_derivative(const bs::Bspline<T>& b, const std::vector<double>& x,
                       const std::vector<int>& ks, double eps) {
  const std::size_t rank = b.rank();
  const std::vector<double> cmax = column_max(b);
  Exact out;
  if (!b.rational()) {
    out.value = exact_hom(b, x, ks);
    for (std::size_t c = 0; c < rank; ++c) {
      out.bound.push_back(2.0 * poly_bound(b, ks, cmax[c], eps));
    }
    return out;
  }
  const auto idx = box(ks);
  struct Entry {
    std::vector<Rational> hom;
    std::vector<Rational> r;
    std::vector<double> err;  // bound on the error of r
    double werr = 0.0;
  };
  std::vector<Entry> table(idx.size());
  const auto find = [&](const std::vector<int>& m) {
    return static_cast<std::size_t>(std::find(idx.begin(), idx.end(), m) - idx.begin());
  };
  for (std::size_t q = 0; q < idx.size(); ++q) {
    table[q].hom = exact_hom(b, x, idx[q]);
    table[q].werr = poly_bound(b, idx[q], cmax[rank], eps);
  }
  const Rational w0 = table[0].hom[rank];
  const double w0d = w0.template convert_to<double>();
  for (std::size_t q = 0; q < idx.size(); ++q) {
    const auto& k = idx[q];
    table[q].r.assign(rank, Rational(0));
    table[q].err.assign(rank, 0.0);
    for (std::size_t c = 0; c < rank; ++c) {
      Rational v = table[q].hom[c];
      double magn = std::fabs(table[q].hom[c].template convert_to<double>());
      double err = poly_bound(b, k, cmax[c], eps);
      double terms = 2.0;
      for (std::size_t p = 1; p <= q; ++p) {
        const auto& i = idx[p];
        if (!leq(i, k)) {
          continue;
        }
        std::vector<int> kmi(k.size());
        double coeff = 1.0;
        for (std::size_t d = 0; d < k.size(); ++d) {
          kmi[d] = k[d] - i[d];
          coeff *= binom(k[d], i[d]);
        }
        const std::size_t r = find(kmi);
        const double wi = table[p].hom[rank].template convert_to<double>();
        const double rd = table[r].r[c].template convert_to<double>();
        v -= Rational(static_cast<long long>(coeff)) * table[p].hom[rank] * table[r].r[c];
        magn += coeff * std::fabs(wi * rd);
        err += coeff * (std::fabs(wi) * table[r].err[c] + table[p].werr * std::fabs(rd));
        terms += 1.0;
      }
      table[q].r[c] = v / w0;
      const double rk = std::fabs(table[q].r[c].template convert_to<double>());
      table[q].err[c] = (err + terms * eps * magn + rk * table[0].werr) / std::fabs(w0d) + eps * rk;
    }
  }
  const std::size_t last = find(ks);
  out.value = table[last].r;
  for (std::size_t c = 0; c < rank; ++c) {
    out.bound.push_back(2.0 * table[last].err[c]);
  }
  return out;
}

/// All points of the Cartesian product of the axes, C order, as rows of a `(n, dim)` array.
template <class P>
std::vector<P> product_points(const std::vector<std::vector<double>>& axes) {
  std::size_t n = 1;
  for (const auto& a : axes) {
    n *= a.size();
  }
  const std::size_t dim = axes.size();
  std::vector<P> pts(n * dim);
  for (std::size_t i = 0; i < n; ++i) {
    std::size_t rest = i;
    for (std::size_t d = dim; d-- > 0;) {
      pts[i * dim + d] = static_cast<P>(axes[d][rest % axes[d].size()]);
      rest /= axes[d].size();
    }
  }
  return pts;
}

/// Evaluate (orders all zero) or differentiate at the product of the axes through the points
/// overload and through the lattice overload, and compare both with the exact map.
template <class T, class P>
void check_against_exact(const bs::Bspline<T>& b, const std::vector<std::vector<double>>& axes_raw,
                         const std::vector<int>& orders) {
  const std::size_t dim = b.dim();
  std::vector<std::vector<double>> axes;
  std::vector<std::vector<P>> lattice_axes;
  for (const auto& a : axes_raw) {
    std::vector<double> q;
    std::vector<P> lp;
    for (const double v : a) {
      q.push_back(as_p<P>(v));
      lp.push_back(static_cast<P>(v));
    }
    axes.push_back(q);
    lattice_axes.push_back(lp);
  }
  const std::vector<P> pts = product_points<P>(axes_raw);
  const std::size_t n = pts.size() / dim;
  const bool value = std::all_of(orders.begin(), orders.end(), [](int o) { return o == 0; });
  const double eps = (std::max)(eps_of<T>(), eps_of<P>());
  const std::size_t rank = b.rank();
  const pantr::quad::PointsLattice<P> lat(lattice_axes);

  NdArray<T> got_pts({n, rank}, pantr::for_overwrite);
  NdArray<T> got_lat({n, rank}, pantr::for_overwrite);
  if (value) {
    // The out overload keeps the weight column: the projection is what the exact map is.
    NdArray<T> full({n, b.cp_size()}, pantr::for_overwrite);
    bs::evaluate<T, P>(b, mdview<const P, 2>(pts.data(), n, dim), full.template view<2>());
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t c = 0; c < rank; ++c) {
        got_pts.data()[i * rank + c] = full.data()[i * b.cp_size() + c];
      }
    }
    bs::evaluate<T, P>(b, lat, full.template view<2>());
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t c = 0; c < rank; ++c) {
        got_lat.data()[i * rank + c] = full.data()[i * b.cp_size() + c];
      }
    }
  } else {
    REQUIRE(bs::evaluate_derivatives<T, P>(b, mdview<const P, 2>(pts.data(), n, dim), orders,
                                           got_pts.template view<2>()));
    REQUIRE(bs::evaluate_derivatives<T, P>(b, lat, orders, got_lat.template view<2>()));
  }
  for (std::size_t i = 0; i < n; ++i) {
    std::vector<double> x(dim);
    for (std::size_t d = 0; d < dim; ++d) {
      x[d] = static_cast<double>(pts[i * dim + d]);
    }
    const Exact want = exact_derivative(b, x, orders, eps);
    CAPTURE(i);
    for (std::size_t c = 0; c < rank; ++c) {
      CAPTURE(c);
      CHECK(oracle::within(static_cast<double>(got_pts.data()[i * rank + c]), want.value[c],
                           want.bound[c]));
      CHECK(oracle::within(static_cast<double>(got_lat.data()[i * rank + c]), want.value[c],
                           want.bound[c]));
    }
  }
}

template <class T>
std::vector<std::vector<double>> axes_of(const bs::Bspline<T>& b) {
  std::vector<std::vector<double>> axes;
  const auto space = b.space();
  for (const auto& s : space.spaces()) {
    axes.push_back(coordinates(s));
  }
  return axes;
}

}  // namespace

// ---- construction and accessors ---------------------------------------------------------------

TEST_CASE("bspline: the copying constructor reads the count and refuses what Python refused") {
  const auto space = make_space<double>({kQuad});  // 6 basis functions
  const std::vector<double> five(5, 1.0);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(bs::Bspline<double>(space, sp(five), false)),
      "The number of control points must be a multiple of the number of basis functions.Got 5 "
      "control points and 6 basis functions.",
      pantr::value_error);
  const std::vector<double> none;
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::Bspline<double>(space, sp(none), false)),
                       "The B-spline must have at least rank one. Got rank 0", pantr::value_error);
  const std::vector<double> six(6, 1.0);
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::Bspline<double>(space, sp(six), true)),
                       "The B-spline must have at least rank one. Got rank 0", pantr::value_error);
  // The count is checked before the rank.
  const std::vector<double> seven(7, 1.0);
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::Bspline<double>(space, sp(seven), true)),
                       "The number of control points must be a multiple of the number of basis "
                       "functions.Got 7 control points and 6 basis functions.",
                       pantr::value_error);
  CHECK(bs::Bspline<double>::cp_size_for(space, 12) == 2);
  CHECK_THROWS_AS(static_cast<void>(bs::Bspline<double>::cp_size_for(space, 13)),
                  pantr::value_error);
}

TEST_CASE("bspline: the shared-buffer and adopting constructors check the rank only") {
  const auto space = make_space<float>({kHat});  // 3 basis functions
  std::shared_ptr<const float[]> data(new float[3]{1.0F, 2.0F, 3.0F});
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::Bspline<float>(space, data, 1, true)),
                       "The B-spline must have at least rank one. Got rank 0", pantr::value_error);
  const bs::Bspline<float> b(space, data, 1, false);
  CHECK(b.control_points().data() == data.get());  // no copy
  CHECK(b.size() == 3);

  NdArray<float> arr({3, 2});
  const float* p = arr.data();
  const bs::Bspline<float> adopted(space, std::move(arr), true);
  CHECK(adopted.control_points().data() == p);
  CHECK(adopted.cp_size() == 2);
  CHECK(adopted.rank() == 1);
  NdArray<float> bad({4});
  CHECK_THROWS_AS(static_cast<void>(bs::Bspline<float>(space, std::move(bad), false)),
                  pantr::value_error);
}

TEST_CASE("bspline: accessors, sharing and the copy of the control points") {
  std::vector<double> cp = control_values(6 * 4, 3);
  const auto space = make_space<double>({kQuad});
  const bs::Bspline<double> b(space, sp(cp), true);
  CHECK(b.dim() == 1);
  CHECK(b.degrees() == std::vector<std::size_t>{2});
  CHECK(b.num_basis(0) == 6);
  CHECK(b.cp_size() == 4);
  CHECK(b.rank() == 3);
  CHECK(b.rational());
  CHECK(b.size() == 24);
  REQUIRE(b.shape().size() == 2);
  CHECK(b.shape()[0] == 6);
  CHECK(b.shape()[1] == 4);
  CHECK(b.control_points().size() == 24);
  // The constructor copied: a later write to the source does not show.
  const double first = cp[0];
  cp[0] = 123.0;
  CHECK(b.control_points()[0] == first);
  // The space is held: it shares its representation with the one the B-spline was built from.
  CHECK(b.space().spaces()[0].rep() == space.spaces()[0].rep());

  // A copy is a reference-count increment of the one representation.
  const long before = b.rep().use_count();
  {
    const bs::Bspline<double> copy = b;
    CHECK(copy.same_rep(b));
    CHECK(b.rep().use_count() == before + 1);
    CHECK(copy.control_points().data() == b.control_points().data());
  }
  CHECK(b.rep().use_count() == before);
  const bs::Bspline<double> other(space, sp(cp), true);
  CHECK_FALSE(other.same_rep(b));
}

TEST_CASE("bspline: a three-direction B-spline has the control-point shape of its space") {
  const std::vector<Dir> dirs = {kQuad, kHat, periodic_dir(1)};
  const auto b = make_bspline<float>(dirs, 3, false);
  CHECK(b.dim() == 3);
  CHECK(b.degrees() == std::vector<std::size_t>{2, 1, 2});
  REQUIRE(b.shape().size() == 4);
  CHECK(b.shape()[0] == 6);
  CHECK(b.shape()[1] == 3);
  CHECK(b.shape()[2] == 4);  // periodic: the stored functions
  CHECK(b.shape()[3] == 3);
}

// ---- evaluation against the exact definition --------------------------------------------------

TEST_CASE_TEMPLATE("bspline: values equal the exact sum over the control points, 1D", Pair, DD,
                   FF) {
  using T = typename Pair::Knot;
  using P = typename Pair::Point;
  for (const Dir& dir : {kQuad, kQuadC0, kCubic, kNonOpen, periodic_dir(1), periodic_dir(0)}) {
    for (const std::size_t cp_size : {std::size_t{1}, std::size_t{3}}) {
      const auto b = make_bspline<T>({dir}, cp_size, false);
      check_against_exact<T, P>(b, axes_of(b), {0});
    }
    const auto r = make_bspline<T>({dir}, 3, true);
    check_against_exact<T, P>(r, axes_of(r), {0});
  }
}

TEST_CASE_TEMPLATE("bspline: values equal the exact sum over the control points, 2D and 3D", Pair,
                   DD, FF) {
  using T = typename Pair::Knot;
  using P = typename Pair::Point;
  const std::vector<std::vector<Dir>> cases = {{kQuad, kHat},
                                               {kNonOpen, periodic_dir(1)},
                                               {periodic_dir(0), kQuadC0},
                                               {kQuad, kNonOpen, periodic_dir(1)}};
  for (const auto& dirs : cases) {
    const auto b = make_bspline<T>(dirs, 2, false, 5);
    check_against_exact<T, P>(b, axes_of(b), std::vector<int>(dirs.size(), 0));
    const auto r = make_bspline<T>(dirs, 3, true, 7);
    check_against_exact<T, P>(r, axes_of(r), std::vector<int>(dirs.size(), 0));
  }
}

TEST_CASE_TEMPLATE("bspline: derivatives equal the exact derivatives of the sum, 1D", Pair, DD,
                   FF) {
  using T = typename Pair::Knot;
  using P = typename Pair::Point;
  for (const Dir& dir : {kQuad, kQuadC0, kCubic, kNonOpen, periodic_dir(1)}) {
    const auto b = make_bspline<T>({dir}, 2, false);
    const auto r = make_bspline<T>({dir}, 3, true);
    // Orders up to one past the degree (identically zero there for the polynomial).
    for (int k = 1; k <= dir.degree + 1; ++k) {
      check_against_exact<T, P>(b, axes_of(b), {k});
      if (k <= 2) {
        check_against_exact<T, P>(r, axes_of(r), {k});
      }
    }
  }
}

TEST_CASE_TEMPLATE("bspline: partial derivatives equal the exact ones, 2D and 3D", Pair, DD, FF) {
  using T = typename Pair::Knot;
  using P = typename Pair::Point;
  const std::vector<Dir> dirs2 = {kQuad, periodic_dir(1)};
  const auto b2 = make_bspline<T>(dirs2, 2, false, 9);
  const auto r2 = make_bspline<T>(dirs2, 3, true, 11);
  for (const std::vector<int>& o : {std::vector<int>{1, 0}, {0, 1}, {1, 1}, {2, 1}, {0, 3}}) {
    check_against_exact<T, P>(b2, axes_of(b2), o);
  }
  for (const std::vector<int>& o : {std::vector<int>{1, 0}, {0, 1}, {1, 1}, {2, 0}}) {
    check_against_exact<T, P>(r2, axes_of(r2), o);
  }
  const std::vector<Dir> dirs3 = {kHat, kNonOpen, kQuadC0};
  const auto b3 = make_bspline<T>(dirs3, 1, false, 13);
  const auto r3 = make_bspline<T>(dirs3, 2, true, 17);
  check_against_exact<T, P>(b3, axes_of(b3), {1, 1, 1});
  check_against_exact<T, P>(r3, axes_of(r3), {1, 1, 0});
}

// ---- a rational curve with known derivatives --------------------------------------------------

TEST_CASE("bspline: the rational quarter circle stays on the circle and its derivatives follow") {
  // The quarter circle as a quadratic NURBS: control points (1, 0), (1, 1), (0, 1) with weights
  // 1, 1/sqrt(2), 1, stored as (w x, w y, w). Every point has |C| = 1, so differentiating
  // |C|^2 = 1 gives C . C' = 0 and C' . C' + C . C'' = 0. At the ends, C'(0) = p (w_1 / w_0)
  // (P_1 - P_0) = 2 (1 / sqrt(2)) (0, 1) = (0, sqrt(2)) and C'(1) = 2 (w_1 / w_2) (P_2 - P_1) =
  // (-sqrt(2), 0) (Piegl and Tiller, the end derivatives of a rational Bezier curve).
  const double s = std::sqrt(0.5);
  const std::vector<double> cp = {1.0, 0.0, 1.0, s, s, s, 0.0, 1.0, 1.0};
  const std::vector<double> knots = {0, 0, 0, 1, 1, 1};
  std::vector<bs::BsplineSpace1D<double>> spaces;
  spaces.emplace_back(sp(knots), 2);
  const bs::Bspline<double> b(bs::BsplineSpace<double>(std::move(spaces)), sp(cp), true);
  CHECK(b.rank() == 2);
  const std::vector<double> t = {0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0};
  const mdview<const double, 2> pts(t.data(), t.size(), 1);
  const auto v = bs::evaluate<double, double>(b, pts);
  const auto d1 = bs::evaluate_derivatives<double, double>(b, pts, std::vector<int>{1});
  const auto d2 = bs::evaluate_derivatives<double, double>(b, pts, std::vector<int>{2});
  REQUIRE(d1.complete);
  REQUIRE(d2.complete);
  // Each identity holds to a few roundings of O(1) quantities (|C'| <= 2 and |C''| <= 8 here, and
  // the formulas are short: 64 eps for the sums, a further factor 64 for the second derivative,
  // whose recurrence and quotient rule are three levels deep).
  const double tol = 64.0 * eps_of<double>();
  for (std::size_t i = 0; i < t.size(); ++i) {
    const double* c = v.data() + 2 * i;
    const double* p = d1.values.data() + 2 * i;
    const double* q = d2.values.data() + 2 * i;
    CAPTURE(t[i]);
    CHECK(std::fabs(c[0] * c[0] + c[1] * c[1] - 1.0) <= tol);
    CHECK(std::fabs(c[0] * p[0] + c[1] * p[1]) <= tol * 4.0);
    CHECK(std::fabs(p[0] * p[0] + p[1] * p[1] + c[0] * q[0] + c[1] * q[1]) <= tol * 64.0);
  }
  const std::size_t last = 2 * (t.size() - 1);
  CHECK(std::fabs(d1.values.data()[0]) <= tol);
  CHECK(std::fabs(d1.values.data()[1] - std::sqrt(2.0)) <= tol * 4.0);
  CHECK(std::fabs(d1.values.data()[last] + std::sqrt(2.0)) <= tol * 4.0);
  CHECK(std::fabs(d1.values.data()[last + 1]) <= tol * 4.0);
  CHECK(std::fabs(v.data()[0] - 1.0) <= tol);
  CHECK(std::fabs(v.data()[1]) <= tol);
  CHECK(std::fabs(v.data()[last]) <= tol);
  CHECK(std::fabs(v.data()[last + 1] - 1.0) <= tol);
}

// ---- the overloads agree, and refuse what they should ------------------------------------------

TEST_CASE("bspline: the out overload keeps the weight, the returning overload drops it") {
  const auto b = make_bspline<double>({kQuad, kHat}, 3, true, 21);
  const std::vector<double> pts = {0.1, 0.2, 0.6, 0.9, 1.0, 0.0};
  const mdview<const double, 2> view(pts.data(), 3, 2);
  NdArray<double> full({3, 3}, pantr::for_overwrite);
  bs::evaluate<double, double>(b, view, full.view<2>());
  const NdArray<double> projected = bs::evaluate<double, double>(b, view);
  REQUIRE(projected.shape()[0] == 3);
  REQUIRE(projected.shape()[1] == 2);
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t c = 0; c < 2; ++c) {
      CHECK(projected.data()[i * 2 + c] == full.data()[i * 3 + c]);
    }
    CHECK(full.data()[i * 3 + 2] > 0.0);  // the weight, left in place
  }
  // The lattice: the shape of the returned array is the lattice's, then the rank.
  const pantr::quad::PointsLattice<double> lat({{0.1, 0.4, 0.9}, {0.2, 0.7}});
  const NdArray<double> grid = bs::evaluate<double, double>(b, lat);
  REQUIRE(grid.rank() == 3);
  CHECK(grid.shape()[0] == 3);
  CHECK(grid.shape()[1] == 2);
  CHECK(grid.shape()[2] == 2);
  const auto nr = make_bspline<double>({kQuad, kHat}, 2, false, 21);
  const NdArray<double> ngrid = bs::evaluate<double, double>(nr, lat);
  REQUIRE(ngrid.rank() == 3);
  CHECK(ngrid.shape()[2] == 2);
  // A point of the lattice is the point evaluation of the same coordinates (C order).
  const std::vector<double> one = {0.4, 0.7};
  const NdArray<double> pt =
      bs::evaluate<double, double>(b, mdview<const double, 2>(one.data(), 1, 2));
  CHECK(grid.data()[(1 * 2 + 1) * 2 + 0] == pt.data()[0]);
  CHECK(grid.data()[(1 * 2 + 1) * 2 + 1] == pt.data()[1]);
}

TEST_CASE("bspline: evaluation refuses a wrong column count, out extents and outside points") {
  const auto b = make_bspline<double>({kQuad, kHat}, 2, false);
  const std::vector<double> pts = {0.1, 0.2, 0.3};
  NdArray<double> out({1, 2});
  CHECK_THROWS_WITH_AS(
      (bs::evaluate<double, double>(b, mdview<const double, 2>(pts.data(), 3, 1), out.view<2>())),
      "pts must have 2 columns", pantr::value_error);
  NdArray<double> bad({2, 2});
  const std::vector<double> ok = {0.1, 0.2};
  CHECK_THROWS_WITH_AS(
      (bs::evaluate<double, double>(b, mdview<const double, 2>(ok.data(), 1, 2), bad.view<2>())),
      "Output array has shape (2, 2), but expected shape (1, 2)", pantr::value_error);
  // A point outside the domain of a direction, and a NaN, are refused with that domain.
  const std::vector<double> outside = {0.1, 1.5};
  CHECK_THROWS_WITH_AS((bs::evaluate<double, double>(
                           b, mdview<const double, 2>(outside.data(), 1, 2), out.view<2>())),
                       "One or more values in pts are outside the knot vector domain (0.0, 1.0)",
                       pantr::value_error);
  const std::vector<double> nan = {std::numeric_limits<double>::quiet_NaN(), 0.5};
  CHECK_THROWS_AS(
      (bs::evaluate<double, double>(b, mdview<const double, 2>(nan.data(), 1, 2), out.view<2>())),
      pantr::value_error);
  const pantr::quad::PointsLattice<double> one_dir({{0.1, 0.2}});
  NdArray<double> lout({2, 2});
  CHECK_THROWS_WITH_AS((bs::evaluate<double, double>(b, one_dir, lout.view<2>())),
                       "Points lattice dimension 1 does not match B-spline dimension 2",
                       pantr::value_error);
}

TEST_CASE("bspline: derivative orders are checked before the points, length first") {
  const auto b = make_bspline<double>({kQuad, kHat}, 1, false);
  const std::vector<double> pts = {0.1, 0.2};
  const mdview<const double, 2> view(pts.data(), 1, 2);
  NdArray<double> out({1, 1});
  CHECK_THROWS_WITH_AS((static_cast<void>(bs::evaluate_derivatives<double, double>(
                           b, view, std::vector<int>{1}, out.view<2>()))),
                       "len(orders) must equal spline.dim (2), got 1", pantr::value_error);
  CHECK_THROWS_WITH_AS((static_cast<void>(bs::evaluate_derivatives<double, double>(
                           b, view, std::vector<int>{1, -1}, out.view<2>()))),
                       "orders[1] must be >= 0, got -1", pantr::value_error);
  // Wrong orders and wrong points together: the orders are reported.
  const std::vector<double> narrow = {0.1};
  CHECK_THROWS_WITH_AS((static_cast<void>(bs::evaluate_derivatives<double, double>(
                           b, mdview<const double, 2>(narrow.data(), 1, 1), std::vector<int>{-1, 0},
                           out.view<2>()))),
                       "orders[0] must be >= 0, got -1", pantr::value_error);
  NdArray<double> bad({2, 1});
  CHECK_THROWS_WITH_AS((static_cast<void>(bs::evaluate_derivatives<double, double>(
                           b, view, std::vector<int>{1, 0}, bad.view<2>()))),
                       "Output array has shape (2, 1), but expected shape (1, 1)",
                       pantr::value_error);
}

TEST_CASE("bspline: a zero divisor in the basis recurrence is a status, not a plausible value") {
  // Degree 2 with the left end repeated four times (p + 2): the span just left of 0 is empty, and a
  // point inside the tolerance band left of 0 divides by zero in the derivative recurrence.
  const std::vector<double> knots = {0, 0, 0, 0, 1, 1, 1};
  std::vector<bs::BsplineSpace1D<double>> spaces;
  spaces.emplace_back(sp(knots), 2);
  const std::vector<double> cp = {1, 2, 3, 4};
  const bs::Bspline<double> b(bs::BsplineSpace<double>(std::move(spaces)), sp(cp), false);
  const std::vector<double> bad = {-1e-17};
  const mdview<const double, 2> view(bad.data(), 1, 1);
  const auto r = bs::evaluate_derivatives<double, double>(b, view, std::vector<int>{1});
  CHECK_FALSE(r.complete);
  NdArray<double> out({1, 1});
  CHECK_FALSE(
      bs::evaluate_derivatives<double, double>(b, view, std::vector<int>{1}, out.view<2>()));
  // Order zero divides by nothing.
  const auto v = bs::evaluate_derivatives<double, double>(b, view, std::vector<int>{0});
  CHECK(v.complete);
}

// ---- threads ----------------------------------------------------------------------------------

namespace {

/// A large evaluation: enough points that the work is parallel (`n * ops >= parallel_min_ops`).
struct Big {
  bs::Bspline<double> b;
  std::vector<double> pts;
  std::size_t n;
};

Big make_big() {
  Big big{make_bspline<double>({kQuad, periodic_dir(1)}, 3, true, 31), {}, 20000};
  big.pts.resize(big.n * 2);
  for (std::size_t i = 0; i < big.n; ++i) {
    const double u = static_cast<double>(i) / static_cast<double>(big.n - 1);
    big.pts[i * 2] = u;
    big.pts[i * 2 + 1] = 1.0 - 0.37 * u;
  }
  return big;
}

NdArray<double> evaluate_big(const Big& big) {
  NdArray<double> out({big.n, 3}, pantr::for_overwrite);
  bs::evaluate<double, double>(big.b, mdview<const double, 2>(big.pts.data(), big.n, 2),
                               out.view<2>());
  return out;
}

NdArray<double> differentiate_big(const Big& big) {
  NdArray<double> out({big.n, 2}, pantr::for_overwrite);
  const bool complete = bs::evaluate_derivatives<double, double>(
      big.b, mdview<const double, 2>(big.pts.data(), big.n, 2), std::vector<int>{1, 1},
      out.view<2>());
  if (!complete) {
    throw std::runtime_error("the derivative recurrence met a zero divisor");
  }
  return out;
}

}  // namespace

TEST_CASE("bspline: a result does not depend on the number of threads") {
  // Bound: zero. Each point is computed from its own basis rows and control points by the same
  // instructions on whichever thread claims its chunk, the chunks are a function of the problem
  // alone, and no loop reduces across chunks; the two runs differ in nothing but who runs them.
  const Big big = make_big();
  const std::size_t saved = pantr::parallel::get_num_threads();
  pantr::parallel::set_num_threads(1);
  const NdArray<double> v1 = evaluate_big(big);
  const NdArray<double> d1 = differentiate_big(big);
  pantr::parallel::set_num_threads(static_cast<std::int64_t>(pantr::parallel::max_threads()));
  const NdArray<double> vk = evaluate_big(big);
  const NdArray<double> dk = differentiate_big(big);
  pantr::parallel::set_num_threads(static_cast<std::int64_t>(saved));
  REQUIRE(v1.size() == vk.size());
  REQUIRE(d1.size() == dk.size());
  std::size_t differing = 0;
  for (std::size_t i = 0; i < v1.size(); ++i) {
    differing += v1.data()[i] == vk.data()[i] ? 0U : 1U;
  }
  for (std::size_t i = 0; i < d1.size(); ++i) {
    differing += d1.data()[i] == dk.data()[i] ? 0U : 1U;
  }
  CHECK(differing == 0);
}

TEST_CASE("bspline: concurrent evaluations of one value from several threads agree") {
  // The value is immutable and shared: each thread reads it, copies it and evaluates on its own
  // output while the others do. Run under ThreadSanitizer this is the test of the shared
  // representation (the control points and the space), and of the pool taking concurrent callers.
  const Big big = make_big();
  const NdArray<double> want_v = evaluate_big(big);
  const NdArray<double> want_d = differentiate_big(big);
  constexpr std::size_t kThreads = 4;
  std::vector<std::size_t> mismatches(kThreads, 0);
  std::vector<std::thread> threads;
  for (std::size_t t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t] {
      const Big mine{big.b, big.pts, big.n};  // a copy of the value: a reference-count increment
      for (int rep = 0; rep < 3; ++rep) {
        const NdArray<double> v = evaluate_big(mine);
        const NdArray<double> d = differentiate_big(mine);
        for (std::size_t i = 0; i < v.size(); ++i) {
          mismatches[t] += v.data()[i] == want_v.data()[i] ? 0U : 1U;
        }
        for (std::size_t i = 0; i < d.size(); ++i) {
          mismatches[t] += d.data()[i] == want_d.data()[i] ? 0U : 1U;
        }
      }
    });
  }
  for (auto& th : threads) {
    th.join();
  }
  for (std::size_t t = 0; t < kThreads; ++t) {
    CHECK(mismatches[t] == 0);
  }
  CHECK(big.b.rep().use_count() == 1);  // every copy was released
}
