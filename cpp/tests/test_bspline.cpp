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

/// `n` equal intervals on [0, 1] (`n` a power of two: dyadic knots), open ends.
Dir open_dir(std::size_t n, int degree) {
  std::vector<double> knots(static_cast<std::size_t>(degree) + 1, 0.0);
  for (std::size_t i = 1; i < n; ++i) {
    knots.push_back(static_cast<double>(i) / static_cast<double>(n));
  }
  knots.resize(knots.size() + static_cast<std::size_t>(degree) + 1, 1.0);
  return {knots, degree};
}

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

/// The number of roundings on the longest path from a term to a value, over the evaluations of
/// `b` that sum its `nloc = prod(degree + 1)` products in different orders.
///
/// The scattered scheme forms a weight by `dim - 1` products, stores it in `T` and adds `nloc - 1`
/// terms: `nloc + dim - 1` roundings. The sum-factorised scheme, per direction, stores a
/// coefficient, multiplies and adds `degree_d` terms: `sum_d (degree_d + 2)`, at most `nloc + dim`
/// since every degree here is at least one (the sum of the orders is at most their product). The
/// bound `nloc + dim` covers both.
template <class T>
double rounding_depth(const bs::Bspline<T>& b) {
  double nloc = 1.0;
  for (const std::size_t p : b.degrees()) {
    nloc *= static_cast<double>(p) + 1.0;
  }
  return nloc + static_cast<double>(b.dim());
}

/// The bound on the difference of two evaluations of `b` at the points `pts` (`n` rows of `dim`)
/// that sum the same products in different orders, per value (`n x rank`).
///
/// The basis values are non-negative, so `S_i = sum_j |w_j| |c_ij|`, the sum of the magnitudes of
/// the terms of coordinate `i`, is the value of the B-spline with the control points `|c|`. With
/// `m = rounding_depth(b)` the number of roundings on the longest path, each evaluation is within
/// `m u S_i` of the exact sum to first order, `u = eps / 2` the unit roundoff (the
/// recursive-summation bound, Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed.,
/// eq. (4.4), applied term by term), so two of them differ by at most `m eps S_i`. A rational value
/// divides a homogeneous coordinate by the weight `H_w`: its error is at most `(dH_i + |R_i| dH_w)
/// / H_w + u |R_i|` with `|R_i| <= S_i / H_w` and `S_w = H_w` (positive weights), that is `(2 m +
/// 1) u S_i / H_w` for each evaluation, and the difference of two is at most `(2 m + 1) eps S_i /
/// H_w`. The basis rows are the same table entries in both, so they add no difference.
template <class T, class P>
std::vector<double> reorder_bound(const bs::Bspline<T>& b, const std::vector<P>& pts,
                                  std::size_t n) {
  std::vector<T> mags(b.control_points().begin(), b.control_points().end());
  for (T& v : mags) {
    v = std::fabs(v);
  }
  const bs::Bspline<T> babs(b.space(), sp(mags), b.rational());
  NdArray<T> full({n, b.cp_size()}, pantr::for_overwrite);
  bs::evaluate<T, P>(babs, mdview<const P, 2>(pts.data(), n, b.dim()), full.template view<2>());
  const double depth = rounding_depth(b);
  const double factor = b.rational() ? 2.0 * depth + 1.0 : depth;
  const double eps = (std::max)(eps_of<T>(), eps_of<P>());
  std::vector<double> bound(n * b.rank());
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t c = 0; c < b.rank(); ++c) {
      bound[i * b.rank() + c] =
          factor * eps * static_cast<double>(full.data()[i * b.cp_size() + c]);
    }
  }
  return bound;
}

/// The number of values of `a` and `b` that differ by more than the bound.
template <class T>
std::size_t exceeding(const NdArray<T>& a, const NdArray<T>& b, const std::vector<double>& bound) {
  REQUIRE(a.size() == b.size());
  REQUIRE(a.size() == bound.size());
  std::size_t count = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double diff =
        std::fabs(static_cast<double>(a.data()[i]) - static_cast<double>(b.data()[i]));
    count += diff <= bound[i] ? 0U : 1U;
  }
  return count;
}

/// The bound, per coordinate, on the difference of two evaluations of a derivative of the
/// non-rational `b` that sum the same products in different orders, at points whose coordinates in
/// direction `d` are `columns[d]`.
///
/// The weight of a local function is a product of one derivative-row entry per direction, so the
/// sum of the magnitudes of the terms is at most `cmax prod_d A_d` with `A_d` the largest, over the
/// coordinates, of the sum of the magnitudes of a row of the table (the entries are signed, so
/// this is not a value of the spline). Two evaluations then differ by at most `m eps cmax prod_d
/// A_d`, by the argument of `reorder_bound`; both read the same table, so its rounding is common.
template <class T, class P>
std::vector<double> derivative_bound(const bs::Bspline<T>& b,
                                     const std::vector<std::vector<P>>& columns,
                                     const std::vector<int>& orders) {
  const auto space = b.space();
  double prod = 1.0;
  for (std::size_t d = 0; d < b.dim(); ++d) {
    const auto r =
        space.spaces()[d].template tabulate_basis_derivatives<P>(sp(columns[d]), orders[d]);
    REQUIRE(r.complete);
    const std::size_t ord = static_cast<std::size_t>(space.spaces()[d].degree()) + 1;
    const std::size_t rows = static_cast<std::size_t>(orders[d]) + 1;
    double largest = 0.0;
    for (std::size_t i = 0; i < columns[d].size(); ++i) {
      double l1 = 0.0;
      for (std::size_t a = 0; a < ord; ++a) {
        l1 += std::fabs(static_cast<double>(r.values.data()[(i * rows + rows - 1) * ord + a]));
      }
      largest = (std::max)(largest, l1);
    }
    prod *= largest;
  }
  const double eps = (std::max)(eps_of<T>(), eps_of<P>());
  std::vector<double> bound = column_max(b);
  for (double& v : bound) {
    v *= rounding_depth(b) * eps * prod;
  }
  return bound;
}

/// A lattice's coordinates: `counts[d]` (at least 2) equally spaced values over the domain of
/// direction `d`.
template <class T, class P>
std::vector<std::vector<P>> uniform_axes(const bs::Bspline<T>& b,
                                         const std::vector<std::size_t>& counts) {
  std::vector<std::vector<P>> axes;
  const auto space = b.space();
  for (std::size_t d = 0; d < counts.size(); ++d) {
    const auto& s = space.spaces()[d];
    const std::size_t m = counts[d];
    const auto [lo, hi] = s.domain();
    std::vector<P> a(m);
    for (std::size_t i = 0; i < m; ++i) {
      const double t = static_cast<double>(i) / static_cast<double>(m - 1);
      a[i] = static_cast<P>(static_cast<double>(lo) +
                            t * (static_cast<double>(hi) - static_cast<double>(lo)));
    }
    axes.push_back(std::move(a));
  }
  return axes;
}

/// The points of the tensor product of `axes`, C order, as rows of a `(n, dim)` array.
template <class P>
std::vector<P> product_rows(const std::vector<std::vector<P>>& axes) {
  std::vector<std::vector<double>> wide;
  for (const auto& a : axes) {
    wide.emplace_back(a.begin(), a.end());
  }
  return product_points<P>(wide);
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

TEST_CASE("bspline: the shared-buffer constructor checks the count first, then the rank") {
  const auto space = make_space<float>({kHat});  // 3 basis functions
  std::shared_ptr<const float[]> data(new float[3]{1.0F, 2.0F, 3.0F});
  CHECK_THROWS_WITH_AS(static_cast<void>(bs::Bspline<float>(space, data, 3, 1, true)),
                       "The B-spline must have at least rank one. Got rank 0", pantr::value_error);
  // The count must be the number of basis functions times the coordinates per control point.
  CHECK_THROWS_WITH_AS(
      static_cast<void>(bs::Bspline<float>(space, data, 3, 2, false)),
      "The buffer holds 3 control-point values, but the space has 3 basis functions of 2 "
      "coordinates each.",
      pantr::value_error);
  CHECK_THROWS_AS(static_cast<void>(bs::Bspline<float>(space, data, 4, 1, false)),
                  pantr::value_error);
  // A product past the largest size matches no count, and is refused rather than wrapped.
  CHECK_THROWS_AS(static_cast<void>(bs::Bspline<float>(
                      space, data, 3, (std::numeric_limits<std::size_t>::max)() / 2, false)),
                  pantr::value_error);
  // A wrong count is reported before a rank below one.
  CHECK_THROWS_WITH_AS(
      static_cast<void>(bs::Bspline<float>(space, data, 4, 1, true)),
      "The buffer holds 4 control-point values, but the space has 3 basis functions of 1 "
      "coordinates each.",
      pantr::value_error);
  const bs::Bspline<float> b(space, data, 3, 1, false);
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
  // A point of the lattice is the point evaluation of the same coordinates (C order). The lattice
  // contracts one direction at a time and the points sum the tensor products, so the two agree
  // to the bound of two summation orders (`reorder_bound`), not bit for bit.
  const std::vector<double> one = {0.4, 0.7};
  const NdArray<double> pt =
      bs::evaluate<double, double>(b, mdview<const double, 2>(one.data(), 1, 2));
  const std::vector<double> bound = reorder_bound<double, double>(b, one, 1);
  for (std::size_t c = 0; c < 2; ++c) {
    CHECK(std::fabs(grid.data()[(1 * 2 + 1) * 2 + c] - pt.data()[c]) <= bound[c]);
  }
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

// ---- the lattice by sum factorisation ---------------------------------------------------------

namespace {

/// Evaluate `b` (values, then derivatives for a non-rational one) on the lattice of `counts`
/// equally spaced coordinates per direction through the lattice and the points overloads, and count
/// the values that differ by more than the bound of two summation orders.
template <class T, class P>
std::size_t lattice_vs_points(const bs::Bspline<T>& b, const std::vector<std::size_t>& counts,
                              const std::vector<std::vector<int>>& orders_list) {
  const auto axes = uniform_axes<T, P>(b, counts);
  const std::vector<P> pts = product_rows<P>(axes);
  const std::size_t n = pts.size() / b.dim();
  const pantr::quad::PointsLattice<P> lat(axes);
  const mdview<const P, 2> view(pts.data(), n, b.dim());
  std::size_t bad = exceeding(bs::evaluate<T, P>(b, view), bs::evaluate<T, P>(b, lat),
                              reorder_bound<T, P>(b, pts, n));
  if (b.rational()) {
    return bad;
  }
  for (const std::vector<int>& orders : orders_list) {
    const auto via_points = bs::evaluate_derivatives<T, P>(b, view, orders);
    const auto via_lattice = bs::evaluate_derivatives<T, P>(b, lat, orders);
    REQUIRE(via_points.complete);
    REQUIRE(via_lattice.complete);
    const std::vector<double> per_column = derivative_bound<T, P>(b, axes, orders);
    std::vector<double> bound(n * b.rank());
    for (std::size_t i = 0; i < bound.size(); ++i) {
      bound[i] = per_column[i % b.rank()];
    }
    bad += exceeding(via_points.values, via_lattice.values, bound);
  }
  return bad;
}

}  // namespace

TEST_CASE_TEMPLATE("bspline: a lattice of many chunks and tiles agrees with the points", Pair, DD,
                   FF) {
  // The contraction cuts a row of `inner` contiguous values into tiles of `chunk_ops / (2 order)`
  // (2731 at degree 2) and a stage into chunks of tiles; the order of the directions is the one
  // that does fewer multiply-adds. Three lattices, each with a stage whose rows span two tiles or
  // more: 40^3 (rows of 2 values, 64000 rows in many chunks), counts (100, 32, 32) with four
  // coordinates (the last-to-first order: its last stage has rows of 4096 values) and counts (2,
  // 40, 40) over directions of 34 functions (the first-to-last order: its first stage has rows of
  // 4624 values).
  using T = typename Pair::Knot;
  using P = typename Pair::Point;
  const std::vector<std::vector<int>> orders = {{1, 0, 1}, {0, 2, 1}, {1, 1, 1}};
  const std::vector<Dir> quads = {kQuad, periodic_dir(1), kNonOpen};
  for (const bool rational : {false, true}) {
    const auto b = make_bspline<T>(quads, rational ? 3 : 2, rational, 41);
    CHECK(lattice_vs_points<T, P>(b, {40, 40, 40}, orders) == 0);
  }
  const auto last_to_first = make_bspline<T>({kQuad, kQuad, kQuad}, 4, false, 43);
  CHECK(lattice_vs_points<T, P>(last_to_first, {100, 32, 32}, orders) == 0);
  const auto wide =
      make_bspline<T>({open_dir(32, 2), open_dir(32, 2), open_dir(32, 2)}, 4, false, 47);
  CHECK(lattice_vs_points<T, P>(wide, {2, 40, 40}, orders) == 0);
}

TEST_CASE_TEMPLATE("bspline: a one-direction lattice and its points agree", Pair, DD, FF) {
  // Both are the same fused loop over the same coordinates, so the bound (`reorder_bound`) is far
  // more than they differ by; it is the bound a result is held to.
  using T = typename Pair::Knot;
  using P = typename Pair::Point;
  for (const Dir& dir : {kQuad, kCubic, periodic_dir(1)}) {
    for (const bool rational : {false, true}) {
      const auto b = make_bspline<T>({dir}, rational ? 3 : 2, rational, 45);
      const auto axes = uniform_axes<T, P>(b, {1000});
      const NdArray<T> via_points =
          bs::evaluate<T, P>(b, mdview<const P, 2>(axes[0].data(), axes[0].size(), 1));
      const NdArray<T> via_lattice = bs::evaluate<T, P>(b, pantr::quad::PointsLattice<P>(axes));
      CHECK(exceeding(via_points, via_lattice, reorder_bound<T, P>(b, axes[0], axes[0].size())) ==
            0);
    }
  }
}

TEST_CASE_TEMPLATE(
    "bspline: the mixed partial of a separable rational surface is the product of its curves'",
    Pair, DD, FF) {
  // R(u, v) = r1(u) r2(v) for the rational curves r1 = (sum N_i w1_i a_i) / (sum N_i w1_i) and r2
  // alike: the surface with homogeneous control points (w1_i a_i w2_j b_j, w1_i w2_j) sums to
  // (sum N_i w1_i a_i) (sum M_j w2_j b_j) over (sum N_i w1_i) (sum M_j w2_j). Its mixed partial is
  // therefore r1'(u) r2'(v), which the one-direction evaluations give with no cross term of the
  // multi-index quotient rule. The values a, b are multiples of 1/16 and the weights 1/2, 1, 3/2,
  // 2, so every control point is exact in float. The bound is the sum of the three first-order
  // bounds of `exact_derivative` (the surface's, and each curve's weighted by the other factor)
  // and the rounding of the product.
  using T = typename Pair::Knot;
  using P = typename Pair::Point;
  struct Factor {
    std::vector<double> a;
    std::vector<double> w;
  };
  const auto factor = [](std::size_t count, unsigned seed) {
    const std::vector<double> v = control_values(count, seed);
    Factor f;
    for (const double x : v) {
      const int k = static_cast<int>(std::lround(x * 64.0));
      f.a.push_back(static_cast<double>(k / 4) / 16.0);
      f.w.push_back(0.5 * static_cast<double>(1 + (k + 64) % 4));
    }
    return f;
  };
  const std::vector<Dir> dirs = {kQuad, kNonOpen};
  const std::size_t n1 = total_basis({kQuad});
  const std::size_t n2 = total_basis({kNonOpen});
  const Factor f1 = factor(n1, 51);
  const Factor f2 = factor(n2, 52);
  const auto curve = [](const Dir& dir, const Factor& f) {
    std::vector<T> cp;
    for (std::size_t i = 0; i < f.a.size(); ++i) {
      cp.push_back(static_cast<T>(f.w[i] * f.a[i]));
      cp.push_back(static_cast<T>(f.w[i]));
    }
    return bs::Bspline<T>(make_space<T>({dir}), sp(cp), true);
  };
  const bs::Bspline<T> c1 = curve(kQuad, f1);
  const bs::Bspline<T> c2 = curve(kNonOpen, f2);
  std::vector<T> cps;
  for (std::size_t i = 0; i < n1; ++i) {
    for (std::size_t j = 0; j < n2; ++j) {
      cps.push_back(static_cast<T>((f1.w[i] * f1.a[i]) * (f2.w[j] * f2.a[j])));
      cps.push_back(static_cast<T>(f1.w[i] * f2.w[j]));
    }
  }
  const bs::Bspline<T> surface(make_space<T>(dirs), sp(cps), true);

  const auto axes = axes_of(surface);
  std::vector<std::vector<double>> d1(2);  // value, bound per coordinate of axis 0
  std::vector<std::vector<double>> d2(2);
  const double eps = (std::max)(eps_of<T>(), eps_of<P>());
  const auto curve_derivatives = [&](const bs::Bspline<T>& c, const std::vector<double>& axis,
                                     std::vector<double>& value, std::vector<double>& bound) {
    std::vector<P> x;
    for (const double v : axis) {
      x.push_back(static_cast<P>(v));
    }
    const auto r = bs::evaluate_derivatives<T, P>(c, mdview<const P, 2>(x.data(), x.size(), 1),
                                                  std::vector<int>{1});
    REQUIRE(r.complete);
    for (std::size_t i = 0; i < x.size(); ++i) {
      value.push_back(static_cast<double>(r.values.data()[i]));
      bound.push_back(exact_derivative(c, {static_cast<double>(x[i])}, {1}, eps).bound[0]);
    }
  };
  curve_derivatives(c1, axes[0], d1[0], d1[1]);
  curve_derivatives(c2, axes[1], d2[0], d2[1]);

  std::vector<std::vector<P>> lattice_axes;
  for (const auto& a : axes) {
    std::vector<P>& axis = lattice_axes.emplace_back();
    for (const auto v : a) {
      axis.push_back(static_cast<P>(v));
    }
  }
  const std::vector<P> pts = product_rows<P>(lattice_axes);
  const std::size_t n = pts.size() / 2;
  const std::vector<int> orders = {1, 1};
  const auto via_points =
      bs::evaluate_derivatives<T, P>(surface, mdview<const P, 2>(pts.data(), n, 2), orders);
  const auto via_lattice =
      bs::evaluate_derivatives<T, P>(surface, pantr::quad::PointsLattice<P>(lattice_axes), orders);
  REQUIRE(via_points.complete);
  REQUIRE(via_lattice.complete);
  std::size_t bad = 0;
  for (std::size_t i = 0; i < axes[0].size(); ++i) {
    for (std::size_t j = 0; j < axes[1].size(); ++j) {
      const std::size_t q = i * axes[1].size() + j;
      const double want = d1[0][i] * d2[0][j];
      const double surface_bound =
          exact_derivative(surface,
                           {static_cast<double>(pts[q * 2]), static_cast<double>(pts[q * 2 + 1])},
                           orders, eps)
              .bound[0];
      const double tol = surface_bound + std::fabs(d2[0][j]) * d1[1][i] +
                         std::fabs(d1[0][i]) * d2[1][j] + d1[1][i] * d2[1][j] +
                         2.0 * eps * std::fabs(want);
      bad += std::fabs(static_cast<double>(via_points.values.data()[q]) - want) <= tol ? 0U : 1U;
      bad += std::fabs(static_cast<double>(via_lattice.values.data()[q]) - want) <= tol ? 0U : 1U;
    }
  }
  CHECK(bad == 0);
}

TEST_CASE("bspline: a refused point leaves the output untouched, in one direction and in two") {
  // The domain is checked before the first write, over the pool for a large input.
  const std::vector<std::size_t> sizes = {7, 100000};
  const double sentinel = 7.25;
  for (const std::size_t n : sizes) {
    const auto b = make_bspline<double>({kCubic}, 3, false);  // domain [0, 4]
    std::vector<double> pts(n);
    for (std::size_t i = 0; i < n; ++i) {
      pts[i] = 4.0 * static_cast<double>(i) / static_cast<double>(n - 1);
    }
    pts[n - 1] = 4.5;
    NdArray<double> out({n, 3}, pantr::for_overwrite);
    std::fill(out.data(), out.data() + out.size(), sentinel);
    CHECK_THROWS_WITH_AS(
        (bs::evaluate<double, double>(b, mdview<const double, 2>(pts.data(), n, 1), out.view<2>())),
        "One or more values in pts are outside the knot vector domain (0.0, 4.0)",
        pantr::value_error);
    CHECK(
        std::all_of(out.data(), out.data() + out.size(), [&](double v) { return v == sentinel; }));
    // A 1D lattice is the same loop; the derivative path checks too.
    const pantr::quad::PointsLattice<double> lat({pts});
    CHECK_THROWS_AS((bs::evaluate<double, double>(b, lat, out.view<2>())), pantr::value_error);
    CHECK(
        std::all_of(out.data(), out.data() + out.size(), [&](double v) { return v == sentinel; }));
    CHECK_THROWS_AS(
        (static_cast<void>(bs::evaluate_derivatives<double, double>(
            b, mdview<const double, 2>(pts.data(), n, 1), std::vector<int>{1}, out.view<2>()))),
        pantr::value_error);
    CHECK(
        std::all_of(out.data(), out.data() + out.size(), [&](double v) { return v == sentinel; }));
  }
  // Two directions: the tabulation checks each direction's coordinates before the contraction.
  const auto b2 = make_bspline<double>({kQuad, kHat}, 2, false);
  const pantr::quad::PointsLattice<double> bad_lat({{0.1, 0.5}, {0.2, 1.5}});
  NdArray<double> out2({4, 2}, pantr::for_overwrite);
  std::fill(out2.data(), out2.data() + out2.size(), sentinel);
  CHECK_THROWS_AS((bs::evaluate<double, double>(b2, bad_lat, out2.view<2>())), pantr::value_error);
  CHECK(
      std::all_of(out2.data(), out2.data() + out2.size(), [&](double v) { return v == sentinel; }));
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
  return bs::evaluate<double, double>(big.b, mdview<const double, 2>(big.pts.data(), big.n, 2));
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

/// Run `f` with one thread and with every thread, and give both results.
template <class F>
auto on_one_and_all_threads(F&& f) {
  const std::size_t saved = pantr::parallel::get_num_threads();
  pantr::parallel::set_num_threads(1);
  auto one = f();
  pantr::parallel::set_num_threads(static_cast<std::int64_t>(pantr::parallel::max_threads()));
  auto all = f();
  pantr::parallel::set_num_threads(static_cast<std::int64_t>(saved));
  return std::make_pair(std::move(one), std::move(all));
}

/// The number of entries of two equally sized arrays that are not equal (compared as values).
std::size_t differing(const NdArray<double>& a, const NdArray<double>& b) {
  REQUIRE(a.size() == b.size());
  std::size_t count = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    count += a.data()[i] == b.data()[i] ? 0U : 1U;
  }
  return count;
}

}  // namespace

TEST_CASE("bspline: a result does not depend on the number of threads") {
  // Exact equality, not a bound. A value is computed from its own table rows and control points by
  // the same instructions on whichever thread claims the chunk that holds it: the chunks are a
  // function of the problem alone, each value is written by one parallel item, and no loop reduces
  // across items. Both runs are in one process, so they execute the same code (the same
  // instruction-set rung, the same contraction) over the same chunks, and the difference between
  // one thread and all is zero by construction. A bound for two summation orders
  // (`reorder_bound`) would admit a dependence on the thread count that this test exists to
  // refuse. Equality across toolchains is a different claim and is not made here.
  const Big big = make_big();
  const auto [v1, vk] = on_one_and_all_threads([&] { return evaluate_big(big); });
  CHECK(differing(v1, vk) == 0);
  // Not a B-spline of the same value: the same space with polynomial control points.
  const auto poly = make_bspline<double>({kQuad, periodic_dir(1)}, 2, false, 33);
  const Big pbig{poly, big.pts, big.n};
  const auto [p1, pk] = on_one_and_all_threads([&] { return differentiate_big(pbig); });
  CHECK(differing(p1, pk) == 0);
  // A rational B-spline, whose derivative is the quotient rule.
  const auto [d1, dk] = on_one_and_all_threads([&] { return differentiate_big(big); });
  CHECK(differing(d1, dk) == 0);

  // The sum-factorised lattice: tiles are cut by the extents alone.
  const auto vol = make_bspline<double>({kQuad, periodic_dir(1), kNonOpen}, 3, true, 35);
  const auto axes = uniform_axes<double, double>(vol, {40, 40, 40});
  const pantr::quad::PointsLattice<double> lat(axes);
  const auto [l1, lk] =
      on_one_and_all_threads([&] { return bs::evaluate<double, double>(vol, lat); });
  CHECK(differing(l1, lk) == 0);
  const auto pvol = make_bspline<double>({kQuad, periodic_dir(1), kNonOpen}, 2, false, 37);
  const auto [ld1, ldk] = on_one_and_all_threads([&] {
    return bs::evaluate_derivatives<double, double>(pvol, lat, std::vector<int>{1, 0, 2}).values;
  });
  CHECK(differing(ld1, ldk) == 0);
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
