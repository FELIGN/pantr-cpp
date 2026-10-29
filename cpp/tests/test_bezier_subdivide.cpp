// Bezier split, slice, boundary, restrict, collapse, compose, reverse, permute and affine
// transform. Oracles, none of them the code under test: the evaluation of the original at the
// reparametrized point (split, slice, restrict, collapse, reverse, permute), the nested evaluation
// outer(inner(t)) and the closed form of a product of two lines (compose), the affine map applied
// to evaluated points (transform), index arithmetic on the control array (the exact copies:
// endpoint slices, reversal, permutation), and exactly representable data (dyadic values under a
// power-of-two matrix).
//
// Tolerances are derived from the unit roundoff `eps` of the storage type and the largest
// control-point magnitude `M`. The bounds used throughout, each an upper bound with slack:
//   - Evaluating a Bezier whose Bernstein weights are convex in each direction: 8 (p_d + 1) eps M
//     per direction, so 8 S eps M with S = sum_d (p_d + 1). A comparison evaluates two Beziers.
//   - One de Casteljau pass (split, slice, one half of a restriction): 12 (p + 1) eps M, the
//     workspace rounding to T at each of p levels plus the double operations; every intermediate
//     is a convex combination, so it stays within M.
//   - A sample point formed in double and rounded to T moves the value by at most p M eps per
//     direction, so P eps M with P = sum_d p_d.
//   - A rational Bezier compares projected values N / W. With weights in [1, 2] and stored
//     (weighted) coordinates of magnitude at most M, an absolute error e in N and W moves N / W
//     by at most e (1 + |N / W|) <= 3 e, plus the rounding of the division: a factor 4 on the
//     polynomial bound.
#include <doctest/doctest.h>

#include <pantr/bezier/compose.hpp>
#include <pantr/bezier/evaluate.hpp>
#include <pantr/bezier/reparametrize.hpp>
#include <pantr/bezier/subdivide.hpp>

#include <algorithm>
#include <array>
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

namespace {

using Bounds = std::optional<std::pair<double, double>>;

template <class T>
double eps_of() {
  return static_cast<double>(std::numeric_limits<T>::epsilon());
}

Bounds bd(double lower, double upper) { return std::make_pair(lower, upper); }

template <class T>
pb::Bezier<T> make_bezier(const std::vector<T>& ctrl, std::vector<std::size_t> shape,
                          bool rational = false) {
  return pb::Bezier<T>(std::span<const T>(ctrl), std::move(shape), rational);
}

template <class T>
std::vector<T> cast_all(const std::vector<double>& v) {
  std::vector<T> out(v.size());
  for (std::size_t i = 0; i < v.size(); ++i) {
    out[i] = static_cast<T>(v[i]);
  }
  return out;
}

struct Lcg {
  std::uint64_t s = 246813579;
  double next() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>(s >> 11) / 4503599627370496.0 - 1.0;  // [-1, 1)
  }
};

// Control points of `rows` points of rank `rank`. Rational: homogeneous (w x, w) with x in
// [-1, 1) and w in [1, 2), so the stored values have magnitude at most 2.
template <class T>
std::vector<T> random_cp(Lcg& rng, std::size_t rows, std::size_t rank, bool rational) {
  std::vector<T> v;
  for (std::size_t r = 0; r < rows; ++r) {
    const double w = rational ? 1.5 + 0.5 * rng.next() : 1.0;
    for (std::size_t c = 0; c < rank; ++c) {
      v.push_back(static_cast<T>(w * rng.next()));
    }
    if (rational) {
      v.push_back(static_cast<T>(w));
    }
  }
  return v;
}

template <class T>
double max_abs(std::span<const T> v) {
  double m = 0.0;
  for (const T x : v) {
    m = (std::max)(m, std::fabs(static_cast<double>(x)));
  }
  return m;
}

template <class T>
pb::Bezier<T> random_bezier(Lcg& rng, const std::vector<std::size_t>& degrees, std::size_t rank,
                            bool rational) {
  std::vector<std::size_t> shape;
  std::size_t rows = 1;
  for (const std::size_t p : degrees) {
    shape.push_back(p + 1);
    rows *= p + 1;
  }
  shape.push_back(rank + (rational ? 1 : 0));
  return make_bezier<T>(random_cp<T>(rng, rows, rank, rational), shape, rational);
}

// S = sum_d (p_d + 1) and P = sum_d p_d of a Bezier.
template <class T>
double sum_p1(const pb::Bezier<T>& b) {
  double s = 0.0;
  for (std::size_t d = 0; d < b.dim(); ++d) {
    s += static_cast<double>(b.degree(d)) + 1.0;
  }
  return s;
}

template <class T>
double sum_p(const pb::Bezier<T>& b) {
  return sum_p1(b) - static_cast<double>(b.dim());
}

// eps M, times 4 for a rational Bezier.
template <class T>
double scale_of(const pb::Bezier<T>& b) {
  return eps_of<T>() * max_abs<T>(b.control_points()) * (b.rational() ? 4.0 : 1.0);
}

// All points of the grid `vals`^dim, row-major, as rows of `dim` values.
template <class T>
std::vector<T> grid(std::size_t dim, const std::vector<double>& vals) {
  std::vector<T> pts;
  std::vector<std::size_t> idx(dim, 0);
  for (;;) {
    for (std::size_t d = 0; d < dim; ++d) {
      pts.push_back(static_cast<T>(vals[idx[d]]));
    }
    bool carry = true;
    for (std::size_t d = dim; d-- > 0 && carry;) {
      if (++idx[d] < vals.size()) {
        carry = false;
      } else {
        idx[d] = 0;
      }
    }
    if (carry) {
      return pts;
    }
  }
}

// x_d -> shift_d + scale_d x_d, formed in double and rounded to T.
template <class T>
std::vector<T> affine_points(const std::vector<T>& pts, std::size_t dim,
                             const std::vector<double>& scale, const std::vector<double>& shift) {
  std::vector<T> out(pts.size());
  for (std::size_t i = 0; i < pts.size(); ++i) {
    const std::size_t d = i % dim;
    out[i] = static_cast<T>(shift[d] + scale[d] * static_cast<double>(pts[i]));
  }
  return out;
}

// Points of a `fixed.size()`-dimensional domain from points of the free directions: the entries of
// `fixed` that hold a value are inserted, the others take the next coordinate of the point.
template <class T>
std::vector<T> embed_points(const std::vector<T>& pts, std::size_t free_dim,
                            const std::vector<std::optional<double>>& fixed) {
  std::vector<T> out;
  for (std::size_t j = 0; j < pts.size() / free_dim; ++j) {
    std::size_t k = 0;
    for (const std::optional<double>& f : fixed) {
      out.push_back(f ? static_cast<T>(*f) : pts[j * free_dim + k++]);
    }
  }
  return out;
}

template <class T>
NdArray<T> eval_at(const pb::Bezier<T>& b, const std::vector<T>& pts) {
  return pb::evaluate<T>(b, pantr::mdview<const T, 2>(pts.data(), pts.size() / b.dim(), b.dim()));
}

template <class T>
double max_diff(const NdArray<T>& a, const NdArray<T>& b) {
  REQUIRE(a.size() == b.size());
  double m = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    m = (std::max)(m,
                   std::fabs(static_cast<double>(a.data()[i]) - static_cast<double>(b.data()[i])));
  }
  return m;
}

template <class T>
bool same_control_points(const pb::Bezier<T>& a, const pb::Bezier<T>& b) {
  if (a.size() != b.size() || a.dim() != b.dim() || a.rational() != b.rational()) {
    return false;
  }
  for (std::size_t d = 0; d <= a.dim(); ++d) {
    if (a.shape()[d] != b.shape()[d]) {
      return false;
    }
  }
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (!(a.control_points()[i] == b.control_points()[i])) {
      return false;
    }
  }
  return true;
}

const std::vector<double>& samples() {
  static const std::vector<double> v = {0.0, 0.125, 0.3, 0.5, 0.8, 1.0};
  return v;
}

const std::vector<double>& coarse_samples() {
  static const std::vector<double> v = {0.0, 0.25, 0.6, 1.0};
  return v;
}

// The cases every reparametrizing test runs over: (degrees, rank, rational).
struct Case {
  std::vector<std::size_t> degrees;
  std::size_t rank;
  bool rational;
};

std::vector<Case> cases() {
  return {{{4}, 2, false}, {{3}, 2, true}, {{3, 2}, 2, false}, {{2, 3}, 2, true}};
}

}  // namespace

TEST_CASE_TEMPLATE("bezier split: the halves are the restrictions to [0, v] and [v, 1]", T, float,
                   double) {
  Lcg rng;
  for (const Case& cs : cases()) {
    const pb::Bezier<T> f = random_bezier<T>(rng, cs.degrees, cs.rank, cs.rational);
    const std::size_t dim = f.dim();
    const std::vector<T> pts = grid<T>(dim, dim == 1 ? samples() : coarse_samples());
    for (std::size_t dir = 0; dir < dim; ++dir) {
      for (const double value : {0.25, 0.5, 0.9}) {
        CAPTURE(dim);
        CAPTURE(dir);
        CAPTURE(value);
        const auto halves = pb::split<T>(f, static_cast<long long>(dir), value);
        const pb::Bezier<T>& left = halves.first;
        const pb::Bezier<T>& right = halves.second;
        CHECK(left.dim() == dim);
        CHECK(right.dim() == dim);
        CHECK(left.rational() == cs.rational);
        CHECK(right.rational() == cs.rational);
        CHECK(left.size() == f.size());
        CHECK(right.size() == f.size());
        // Eval of the original (8 S), of the half (8 S), one pass (12 (p_dir + 1)), the sample
        // point (P).
        const double tol =
            scale_of(f) *
            (16.0 * sum_p1(f) + 12.0 * (static_cast<double>(f.degree(dir)) + 1.0) + sum_p(f));
        std::vector<double> scale_l(dim, 1.0);
        std::vector<double> shift_l(dim, 0.0);
        std::vector<double> scale_r(dim, 1.0);
        std::vector<double> shift_r(dim, 0.0);
        scale_l[dir] = value;
        scale_r[dir] = 1.0 - value;
        shift_r[dir] = value;
        const NdArray<T> want_l = eval_at(f, affine_points(pts, dim, scale_l, shift_l));
        const NdArray<T> want_r = eval_at(f, affine_points(pts, dim, scale_r, shift_r));
        const NdArray<T> got_l = eval_at(left, pts);
        const NdArray<T> got_r = eval_at(right, pts);
        const double err_l = max_diff(got_l, want_l);
        const double err_r = max_diff(got_r, want_r);
        CHECK(err_l <= tol);
        CHECK(err_r <= tol);
      }
    }
  }
}

TEST_CASE_TEMPLATE("bezier split: the halves of a curve meet at f(value)", T, float, double) {
  Lcg rng;
  for (const bool rational : {false, true}) {
    const pb::Bezier<T> f = random_bezier<T>(rng, {5}, 2, rational);
    const double value = 0.37;
    const auto halves = pb::split<T>(f, 0, value);
    const pb::Bezier<T>& left = halves.first;
    const pb::Bezier<T>& right = halves.second;
    // The last control point of the left half and the first of the right half are the apex of one
    // de Casteljau triangle: the same stored value.
    const std::size_t cp = f.cp_size();
    const std::size_t last = f.size() - cp;
    bool apex_equal = true;
    for (std::size_t c = 0; c < cp; ++c) {
      apex_equal = apex_equal && (left.control_points()[last + c] == right.control_points()[c]);
    }
    CHECK(apex_equal);
    // The end points of each half are those of the original.
    bool ends_equal = true;
    for (std::size_t c = 0; c < cp; ++c) {
      ends_equal = ends_equal && (left.control_points()[c] == f.control_points()[c]) &&
                   (right.control_points()[last + c] == f.control_points()[last + c]);
    }
    CHECK(ends_equal);
    const std::vector<T> at_value = cast_all<T>({value});
    const std::vector<T> one = cast_all<T>({1.0});
    const std::vector<T> zero = cast_all<T>({0.0});
    const NdArray<T> fv = eval_at(f, at_value);
    const NdArray<T> l1 = eval_at(left, one);
    const NdArray<T> r0 = eval_at(right, zero);
    const double tol = scale_of(f) * (16.0 * sum_p1(f) + 12.0 * sum_p1(f) + sum_p(f));
    const double err_l = max_diff(l1, fv);
    const double err_r = max_diff(r0, fv);
    CHECK(err_l <= tol);
    CHECK(err_r <= tol);
  }
}

TEST_CASE_TEMPLATE("bezier slice: a surface slice is the section of the surface", T, float,
                   double) {
  Lcg rng;
  for (const bool rational : {false, true}) {
    // A volume of degrees (2, 1, 3): slice each axis.
    const pb::Bezier<T> f = random_bezier<T>(rng, {2, 1, 3}, 2, rational);
    for (std::size_t axis = 0; axis < 3; ++axis) {
      for (const double value : {0.2, 0.75}) {
        CAPTURE(axis);
        CAPTURE(value);
        const pb::Bezier<T> s = pb::slice<T>(f, static_cast<long long>(axis), value);
        CHECK(s.dim() == 2);
        CHECK(s.rational() == rational);
        CHECK(s.cp_size() == f.cp_size());
        const std::vector<T> pts = grid<T>(2, coarse_samples());
        std::vector<std::optional<double>> fixed(3);
        fixed[axis] = value;
        const NdArray<T> want = eval_at(f, embed_points(pts, 2, fixed));
        const NdArray<T> got = eval_at(s, pts);
        // Eval of both (8 S each), one pass in `axis`, the value is exact in double but the point
        // handed to the evaluation is rounded to T: P.
        const double tol =
            scale_of(f) * (8.0 * sum_p1(f) + 8.0 * sum_p1(s) +
                           12.0 * (static_cast<double>(f.degree(axis)) + 1.0) + sum_p(f));
        const double err = max_diff(got, want);
        CHECK(err <= tol);
      }
    }
  }
}

TEST_CASE_TEMPLATE("bezier slice: the end values copy the first or last slab", T, float, double) {
  Lcg rng;
  // A surface of shape (4, 3, 2): element (i, j, c) of the control array is cp[(i * 3 + j) * 2 +
  // c].
  const pb::Bezier<T> f = random_bezier<T>(rng, {3, 2}, 2, false);
  const auto cp = f.control_points();
  const pb::Bezier<T> u0 = pb::slice<T>(f, 0, 0.0);
  const pb::Bezier<T> u1 = pb::slice<T>(f, 0, 1.0);
  const pb::Bezier<T> v0 = pb::slice<T>(f, 1, 0.0);
  const pb::Bezier<T> v1 = pb::slice<T>(f, 1, 1.0);
  CHECK(u0.dim() == 1);
  CHECK(u0.degree(0) == 2);
  CHECK(v0.degree(0) == 3);
  bool equal = true;
  for (std::size_t j = 0; j < 3; ++j) {
    for (std::size_t c = 0; c < 2; ++c) {
      equal = equal && (u0.control_points()[j * 2 + c] == cp[(0 * 3 + j) * 2 + c]);
      equal = equal && (u1.control_points()[j * 2 + c] == cp[(3 * 3 + j) * 2 + c]);
    }
  }
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t c = 0; c < 2; ++c) {
      equal = equal && (v0.control_points()[i * 2 + c] == cp[(i * 3 + 0) * 2 + c]);
      equal = equal && (v1.control_points()[i * 2 + c] == cp[(i * 3 + 2) * 2 + c]);
    }
  }
  CHECK(equal);
  // A volume, the middle axis: element (i, j, k, c) is cp[((i * 3 + j) * 2 + k) * 1 + c] for shape
  // (2, 3, 2, 1).
  const pb::Bezier<T> g = random_bezier<T>(rng, {1, 2, 1}, 1, false);
  const pb::Bezier<T> m1 = pb::slice<T>(g, 1, 1.0);
  CHECK(m1.dim() == 2);
  bool mid_equal = true;
  for (std::size_t i = 0; i < 2; ++i) {
    for (std::size_t k = 0; k < 2; ++k) {
      mid_equal = mid_equal &&
                  (m1.control_points()[i * 2 + k] == g.control_points()[((i * 3 + 2) * 2 + k)]);
    }
  }
  CHECK(mid_equal);
  // The boundary is the slice at an end.
  const pb::Bezier<T> b_u1 = pb::boundary<T>(f, 0, 1);
  const pb::Bezier<T> b_v0 = pb::boundary<T>(f, 1, 0);
  CHECK(same_control_points<T>(b_u1, u1));
  CHECK(same_control_points<T>(b_v0, v0));
}

TEST_CASE_TEMPLATE("bezier slice: a curve slices to a point", T, float, double) {
  Lcg rng;
  const pb::Bezier<T> f = random_bezier<T>(rng, {4}, 2, false);
  const pb::Bezier<T> r = random_bezier<T>(rng, {3}, 2, true);
  for (const pb::Bezier<T>* b : {&f, &r}) {
    const std::size_t rank = b->rank();
    const std::size_t cp = b->cp_size();
    const std::size_t last = b->size() - cp;
    // The end values: the first or last control point, projected by the weight when rational.
    const NdArray<T> p0 = pb::slice_point<T>(*b, 0, 0.0);
    const NdArray<T> p1 = pb::slice_point<T>(*b, 0, 1.0);
    const NdArray<T> q0 = pb::boundary_point<T>(*b, 0, 0);
    const NdArray<T> q1 = pb::boundary_point<T>(*b, 0, 1);
    REQUIRE(p0.rank() == 1);
    CHECK(p0.shape()[0] == rank);
    bool equal = true;
    for (std::size_t c = 0; c < rank; ++c) {
      const T w0 = b->rational() ? b->control_points()[rank] : static_cast<T>(1);
      const T w1 = b->rational() ? b->control_points()[last + rank] : static_cast<T>(1);
      const T want0 = b->rational() ? b->control_points()[c] / w0 : b->control_points()[c];
      const T want1 =
          b->rational() ? b->control_points()[last + c] / w1 : b->control_points()[last + c];
      equal = equal && (p0.data()[c] == want0) && (p1.data()[c] == want1) &&
              (q0.data()[c] == want0) && (q1.data()[c] == want1);
    }
    CHECK(equal);
    // An interior value against the evaluation: eval (8 S) plus one pass (12 S) plus the point.
    const double tol = scale_of(*b) * (8.0 * sum_p1(*b) + 12.0 * sum_p1(*b) + sum_p(*b));
    for (const double value : {0.1, 0.5, 0.93}) {
      const NdArray<T> pt = pb::slice_point<T>(*b, 0, value);
      const NdArray<T> want = eval_at(*b, cast_all<T>({value}));
      const double err = max_diff(pt, want);
      CAPTURE(value);
      CHECK(err <= tol);
    }
  }
}

TEST_CASE_TEMPLATE("bezier restrict: the restriction is the map on the sub-box", T, float, double) {
  Lcg rng;
  // (0.3, 0.95) takes the upper bound first, (0.05, 0.4) and (0.2, 0.7) the lower bound first
  // (restrict_1d_kernel compares |upper| with |lower - 1|); (0.0, 0.6) has a zero lower bound.
  const std::vector<std::pair<double, double>> boxes = {
      {0.3, 0.95}, {0.05, 0.4}, {0.2, 0.7}, {0.0, 0.6}};
  for (const Case& cs : cases()) {
    const pb::Bezier<T> f = random_bezier<T>(rng, cs.degrees, cs.rank, cs.rational);
    const std::size_t dim = f.dim();
    const std::vector<T> pts = grid<T>(dim, dim == 1 ? samples() : coarse_samples());
    for (const auto& box : boxes) {
      const double lower = box.first;
      const double upper = box.second;
      // Restrict the first direction; a second direction takes the next box and the third none.
      std::vector<Bounds> bounds(dim);
      std::vector<double> scale(dim, 1.0);
      std::vector<double> shift(dim, 0.0);
      bounds[0] = bd(lower, upper);
      scale[0] = upper - lower;
      shift[0] = lower;
      double restricted_p1 = static_cast<double>(f.degree(0)) + 1.0;
      if (dim == 2) {
        bounds[1] = bd(0.1, 0.8);
        scale[1] = 0.8 - 0.1;
        shift[1] = 0.1;
        restricted_p1 += static_cast<double>(f.degree(1)) + 1.0;
      }
      CAPTURE(dim);
      CAPTURE(lower);
      CAPTURE(upper);
      const pb::Bezier<T> r = pb::restrict<T>(f, std::span<const Bounds>(bounds));
      CHECK(r.dim() == dim);
      CHECK(r.rational() == cs.rational);
      CHECK(r.size() == f.size());
      // Eval of both (8 S each), two passes per restricted direction (24 (p + 1)), the sample
      // point and the pass parameters lower / upper, (upper - lower) / (1 - lower): 8 P.
      const double tol = scale_of(f) * (16.0 * sum_p1(f) + 24.0 * restricted_p1 + 8.0 * sum_p(f));
      const NdArray<T> want = eval_at(f, affine_points(pts, dim, scale, shift));
      const NdArray<T> got = eval_at(r, pts);
      const double err = max_diff(got, want);
      CHECK(err <= tol);
    }
    if (dim == 2) {
      // Only the second direction restricted, the first given as the full domain.
      const std::vector<Bounds> bounds = {bd(0.0, 1.0), bd(0.2, 0.7)};
      const pb::Bezier<T> r = pb::restrict<T>(f, std::span<const Bounds>(bounds));
      const std::vector<Bounds> only_second = {std::nullopt, bd(0.2, 0.7)};
      const pb::Bezier<T> r2 = pb::restrict<T>(f, std::span<const Bounds>(only_second));
      CHECK(same_control_points<T>(r, r2));
    }
  }
}

TEST_CASE_TEMPLATE("bezier collapse: the curve is the section through the fixed values", T, float,
                   double) {
  Lcg rng;
  for (const bool rational : {false, true}) {
    const pb::Bezier<T> f = random_bezier<T>(rng, {3, 2, 4}, 2, rational);
    const std::vector<T> ts = grid<T>(1, samples());
    for (std::size_t axis = 0; axis < 3; ++axis) {
      const std::vector<T> values = cast_all<T>(axis == 0   ? std::vector<double>{0.3, 0.7}
                                                : axis == 1 ? std::vector<double>{0.9, 0.2}
                                                            : std::vector<double>{0.4, 0.6});
      CAPTURE(axis);
      const pb::Bezier<T> c =
          pb::collapse_along_axis<T>(f, static_cast<long long>(axis), std::span<const T>(values));
      CHECK(c.dim() == 1);
      CHECK(c.degree(0) == f.degree(axis));
      CHECK(c.cp_size() == f.cp_size());
      CHECK(c.rational() == rational);
      std::vector<std::optional<double>> fixed(3);
      std::size_t k = 0;
      double collapsed_p1 = 0.0;
      for (std::size_t d = 0; d < 3; ++d) {
        if (d != axis) {
          fixed[d] = static_cast<double>(values[k++]);
          collapsed_p1 += static_cast<double>(f.degree(d)) + 1.0;
        }
      }
      const NdArray<T> want = eval_at(f, embed_points(ts, 1, fixed));
      const NdArray<T> got = eval_at(c, ts);
      // Eval of the volume (8 S) and of the curve (8 (p_axis + 1)); each collapsed direction
      // contracts p + 1 terms whose Bernstein values carry a relative error of about 2 (p + 1) eps
      // and whose sum adds p + 1 roundings: 4 (p + 1) eps M.
      const double tol = scale_of(f) * (8.0 * sum_p1(f) + 8.0 * sum_p1(c) + 4.0 * collapsed_p1);
      const double err = max_diff(got, want);
      CHECK(err <= tol);
    }
  }
}

TEST_CASE_TEMPLATE("bezier compose: the product of two lines against its closed form", T, float,
                   double) {
  // f(u, v) = u v has the control array (0, 0, 0, 1) over (2, 2); with the lines g_0 = (a, b) and
  // g_1 = (c, d) the composition is the quadratic g_0 g_1 with the Bernstein coefficients
  // (a c, (a d + b c) / 2, b d).
  const double a = 0.2;
  const double b = 0.8;
  const double c = 0.1;
  const double d = 0.9;
  const pb::Bezier<T> outer = make_bezier<T>(cast_all<T>({0.0, 0.0, 0.0, 1.0}), {2, 2, 1});
  const pb::Bezier<T> inner = make_bezier<T>(cast_all<T>({a, c, b, d}), {2, 2});
  const pb::Bezier<T> h = pb::compose<T>(outer, inner);
  REQUIRE(h.dim() == 1);
  CHECK(h.degree(0) == 2);
  CHECK(h.cp_size() == 1);
  CHECK_FALSE(h.rational());
  // Each coefficient is formed from at most four stored operations on values of magnitude below
  // one: 8 eps (every product, binomial and sum rounds once; the coefficients of the products of
  // the factors a, 1 - a, ... are below one).
  const double tol = 8.0 * eps_of<T>();
  const std::array<double, 3> want = {a * c, (a * d + b * c) / 2.0, b * d};
  for (std::size_t i = 0; i < 3; ++i) {
    const double err = std::fabs(static_cast<double>(h.control_points()[i]) - want[i]);
    CAPTURE(i);
    CHECK(err <= tol);
  }
}

namespace {

// The bound on a composition against the nested evaluation, for an outer of degrees `deg_f`
// (S_f = sum, m_max the largest) and an inner of degrees `n` (the largest n_max), outer magnitude
// `M_f`. R = S_f n_max is the largest composed degree, K = S_f + dim_f the number of products in
// the chain of one Bernstein function, N_pts the number of control points of the outer.
//   - The Bernstein functions of the inner map have coefficients in [0, 1] and every product
//     of them sums non-negative terms, so each product coefficient carries a relative error of at
//     most (R + 4) eps (one rounding per stored partial sum, the division and the binomials), and
//     the chain accumulates K of them: K (R + 4) eps, against a total weight of 1.
//   - Adding the N_pts terms coef * basis in T: N_pts eps M_f more.
//   - Evaluating the composition (8 (R + 1) eps M_f), and evaluating the outer at the evaluated
//     inner point (8 (m + 1) eps M_f per direction, summed: 8 (S_f + dim_f) eps M_f), whose
//     argument has an error 8 (n_max + 1) eps that the outer, of Lipschitz constant at most 2 M_f
//     S_f, amplifies by 16 S_f (n_max + 1) eps M_f.
template <class T>
double compose_tol(const pb::Bezier<T>& outer, const pb::Bezier<T>& inner) {
  double s_f = 0.0;
  for (std::size_t d = 0; d < outer.dim(); ++d) {
    s_f += static_cast<double>(outer.degree(d));
  }
  double n_max = 0.0;
  for (std::size_t d = 0; d < inner.dim(); ++d) {
    n_max = (std::max)(n_max, static_cast<double>(inner.degree(d)));
  }
  const double dim_f = static_cast<double>(outer.dim());
  const double r = s_f * n_max;
  const double k = s_f + dim_f;
  const double n_pts = static_cast<double>(outer.size()) / static_cast<double>(outer.cp_size());
  const double m_f = max_abs<T>(outer.control_points());
  return eps_of<T>() * m_f *
         (k * (r + 4.0) + n_pts + 8.0 * (r + 1.0) + 8.0 * (s_f + dim_f) +
          16.0 * s_f * (n_max + 1.0));
}

template <class T>
void check_compose_against_nested(const pb::Bezier<T>& outer, const pb::Bezier<T>& inner) {
  const pb::Bezier<T> h = pb::compose<T>(outer, inner);
  REQUIRE(h.dim() == inner.dim());
  CHECK(h.cp_size() == outer.rank());
  CHECK_FALSE(h.rational());
  double s_f = 0.0;
  for (std::size_t d = 0; d < outer.dim(); ++d) {
    s_f += static_cast<double>(outer.degree(d));
  }
  for (std::size_t s = 0; s < inner.dim(); ++s) {
    const double want_degree = s_f * static_cast<double>(inner.degree(s));
    const double got_degree = static_cast<double>(h.degree(s));
    CHECK(got_degree == want_degree);
  }
  const std::vector<T> pts = grid<T>(inner.dim(), inner.dim() == 1 ? samples() : coarse_samples());
  const NdArray<T> mid = eval_at(inner, pts);
  const std::vector<T> mid_pts(mid.data(), mid.data() + mid.size());
  const NdArray<T> want = eval_at(outer, mid_pts);
  const NdArray<T> got = eval_at(h, pts);
  const double tol = compose_tol(outer, inner);
  const double err = max_diff(got, want);
  CHECK(err <= tol);
}

}  // namespace

TEST_CASE_TEMPLATE("bezier compose: against the nested evaluation, 1D and nD inner maps", T, float,
                   double) {
  Lcg rng;
  // The inner control points lie in [0, 1], so the composition stays in the outer's domain.
  auto unit_inner = [&rng](const std::vector<std::size_t>& degrees, std::size_t rank) {
    std::vector<std::size_t> shape;
    std::size_t rows = 1;
    for (const std::size_t n : degrees) {
      shape.push_back(n + 1);
      rows *= n + 1;
    }
    shape.push_back(rank);
    std::vector<T> v(rows * rank);
    for (T& x : v) {
      x = static_cast<T>(0.5 * (rng.next() + 1.0));
    }
    return make_bezier<T>(v, shape);
  };
  // 1D outer of degree 3 with a 1D inner of degree 2.
  check_compose_against_nested<T>(random_bezier<T>(rng, {3}, 2, false), unit_inner({2}, 1));
  // A surface outer (degrees (2, 1)) with a curve inner of rank 2.
  check_compose_against_nested<T>(random_bezier<T>(rng, {2, 1}, 2, false), unit_inner({2}, 2));
  // A degree-0 direction in the outer: degrees (0, 2).
  check_compose_against_nested<T>(random_bezier<T>(rng, {0, 2}, 1, false), unit_inner({2}, 2));
  // A 1D outer with a surface inner (degrees (1, 2), rank 1), and a surface outer with a surface
  // inner (degrees (1, 1) in both), which take the tensor-product formula.
  check_compose_against_nested<T>(random_bezier<T>(rng, {3}, 2, false), unit_inner({1, 2}, 1));
  check_compose_against_nested<T>(random_bezier<T>(rng, {1, 1}, 1, false), unit_inner({1, 1}, 2));
}

TEST_CASE_TEMPLATE("bezier compose: the exempt and the checked envelope", T, float, double) {
  // A 1D outer of degree one forms no product, so a degree-70 inner is composed. The outer is the
  // identity f(x) = x, control points (0, 1), and the inner the degree-70 identity, whose control
  // points are i / 70: the composition is 0 (1 - g) + 1 g = g, each coefficient the sum 0 + 1 g_i
  // of exact terms.
  std::vector<double> ramp(71);
  for (std::size_t i = 0; i < ramp.size(); ++i) {
    ramp[i] = static_cast<double>(i) / 70.0;
  }
  const pb::Bezier<T> ident = make_bezier<T>(cast_all<T>({0.0, 1.0}), {2, 1});
  const pb::Bezier<T> ramp70 = make_bezier<T>(cast_all<T>(ramp), {71, 1});
  const pb::Bezier<T> h = pb::compose<T>(ident, ramp70);
  CHECK(h.degree(0) == 70);
  CHECK(same_control_points<T>(h, ramp70));

  const std::string tail =
      " needs binomial coefficients up to C(66, k), beyond the largest upper index 61 that pantr's "
      "exact-integer binomial kernel can compute without an int64 overflow. Past that the "
      "coefficients wrap silently and the result is corrupted rather than merely inaccurate.";
  // A 1D outer of degree 6 with a 1D inner of degree 11 asks for C(66, k).
  const pb::Bezier<T> outer6 = make_bezier<T>(cast_all<T>({0, 1, 2, 3, 4, 5, 6}), {7, 1});
  const pb::Bezier<T> inner11 = make_bezier<T>(
      cast_all<T>({0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0}), {12, 1});
  const std::string want66 = "Composition to degree 66 with a 1D inner Bézier" + tail;
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::compose<T>(outer6, inner11)), want66.c_str(),
                       pantr::value_error);
  // A degree-(1, 1) surface is not exempt: degree 31 gives 62, degree 30 gives 60.
  const pb::Bezier<T> bilinear = make_bezier<T>(cast_all<T>({0.0, 0.0, 0.0, 1.0}), {2, 2, 1});
  auto line_pair = [](std::size_t degree) {
    std::vector<double> v;
    for (std::size_t i = 0; i <= degree; ++i) {
      const double s = static_cast<double>(i) / static_cast<double>(degree);
      v.push_back(s);
      v.push_back(1.0 - s);
    }
    return make_bezier<T>(cast_all<T>(v), {degree + 1, 2});
  };
  const std::string tail62 =
      " needs binomial coefficients up to C(62, k), beyond the largest upper index 61 that pantr's "
      "exact-integer binomial kernel can compute without an int64 overflow. Past that the "
      "coefficients wrap silently and the result is corrupted rather than merely inaccurate.";
  const std::string want62 = "Composition to degree 62 with a 1D inner Bézier" + tail62;
  const pb::Bezier<T> pair31 = line_pair(31);
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::compose<T>(bilinear, pair31)), want62.c_str(),
                       pantr::value_error);
  const pb::Bezier<T> pair30 = line_pair(30);
  const pb::Bezier<T> ok = pb::compose<T>(bilinear, pair30);
  CHECK(ok.degree(0) == 60);
}

TEST_CASE_TEMPLATE("bezier compose: the checks and their messages", T, float, double) {
  const pb::Bezier<T> curve = make_bezier<T>(cast_all<T>({0.0, 1.0}), {2, 1});
  const pb::Bezier<T> rat = make_bezier<T>(cast_all<T>({0.0, 1.0, 1.0, 1.0}), {2, 2}, true);
  const pb::Bezier<T> line2 = make_bezier<T>(cast_all<T>({0.0, 0.0, 1.0, 1.0}), {2, 2});
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::compose<T>(rat, curve)),
                       "Composition is not supported for rational Béziers (outer is rational).",
                       pantr::type_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::compose<T>(curve, rat)),
                       "Composition is not supported for rational Béziers (inner is rational).",
                       pantr::type_error);
  // Both rational: the outer is reported first.
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::compose<T>(rat, rat)),
                       "Composition is not supported for rational Béziers (outer is rational).",
                       pantr::type_error);
  // The inner has rank 1, the outer dimension 2; and the converse.
  const pb::Bezier<T> surf = make_bezier<T>(cast_all<T>({0.0, 0.0, 0.0, 1.0}), {2, 2, 1});
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::compose<T>(surf, curve)),
                       "Inner Bézier rank (1) must equal outer Bézier parametric dimension (2).",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::compose<T>(curve, line2)),
                       "Inner Bézier rank (2) must equal outer Bézier parametric dimension (1).",
                       pantr::value_error);
}

TEST_CASE_TEMPLATE("bezier reversed: t becomes 1 - t in the chosen direction", T, float, double) {
  Lcg rng;
  for (const Case& cs : cases()) {
    const pb::Bezier<T> f = random_bezier<T>(rng, cs.degrees, cs.rank, cs.rational);
    const std::size_t dim = f.dim();
    // Dyadic samples, so that 1 - t is exact in T.
    const std::vector<double> dyadic = {0.0, 0.125, 0.5, 0.75, 1.0};
    const std::vector<T> pts = grid<T>(dim, dyadic);
    for (std::size_t dir = 0; dir < dim; ++dir) {
      CAPTURE(dim);
      CAPTURE(dir);
      const pb::Bezier<T> r = pb::reversed<T>(f, static_cast<long long>(dir));
      CHECK(r.rational() == cs.rational);
      std::vector<double> scale(dim, 1.0);
      std::vector<double> shift(dim, 0.0);
      scale[dir] = -1.0;
      shift[dir] = 1.0;
      const NdArray<T> want = eval_at(f, affine_points(pts, dim, scale, shift));
      const NdArray<T> got = eval_at(r, pts);
      const double tol = scale_of(f) * 16.0 * sum_p1(f);
      const double err = max_diff(got, want);
      CHECK(err <= tol);
      // Twice reversed is the original.
      const pb::Bezier<T> back = pb::reversed<T>(r, static_cast<long long>(dir));
      CHECK(same_control_points<T>(back, f));
    }
  }
  // A surface of shape (3, 2, 2): the control point (i, j) of the reversal in direction 0 is the
  // point (2 - i, j) of the original.
  const pb::Bezier<T> g = random_bezier<T>(rng, {2, 1}, 2, false);
  const pb::Bezier<T> rg = pb::reversed<T>(g, 0);
  bool equal = true;
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 2; ++j) {
      for (std::size_t c = 0; c < 2; ++c) {
        equal = equal && (rg.control_points()[(i * 2 + j) * 2 + c] ==
                          g.control_points()[((2 - i) * 2 + j) * 2 + c]);
      }
    }
  }
  CHECK(equal);
}

TEST_CASE_TEMPLATE("bezier permuted: directions are reordered", T, float, double) {
  Lcg rng;
  // A surface: the new direction 0 is the old direction 1, so r(t0, t1) = f(t1, t0).
  for (const bool rational : {false, true}) {
    const pb::Bezier<T> f = random_bezier<T>(rng, {3, 2}, 2, rational);
    const std::vector<long long> swap01 = {1, 0};
    const pb::Bezier<T> r = pb::permuted<T>(f, std::span<const long long>(swap01));
    CHECK(r.degree(0) == 2);
    CHECK(r.degree(1) == 3);
    CHECK(r.rational() == rational);
    const std::vector<T> pts = grid<T>(2, coarse_samples());
    // The swapped sample points: (t1, t0).
    std::vector<T> swapped(pts.size());
    for (std::size_t j = 0; j < pts.size() / 2; ++j) {
      swapped[2 * j] = pts[2 * j + 1];
      swapped[2 * j + 1] = pts[2 * j];
    }
    const NdArray<T> want_swapped = eval_at(f, swapped);
    const NdArray<T> got = eval_at(r, pts);
    const double tol = scale_of(f) * 16.0 * sum_p1(f);
    const double err = max_diff(got, want_swapped);
    CHECK(err <= tol);
  }
  // A volume of shape (2, 3, 4, 2) and the permutation [1, 2, 0]: the new direction 0 is the old
  // 1, the new 1 the old 2 and the new 2 the old 0, so r[a, b, c] = f[c, a, b] and the shape is
  // (3, 4, 2, 2).
  const pb::Bezier<T> v = random_bezier<T>(rng, {1, 2, 3}, 2, false);
  const std::vector<long long> perm = {1, 2, 0};
  const pb::Bezier<T> rv = pb::permuted<T>(v, std::span<const long long>(perm));
  REQUIRE(rv.dim() == 3);
  CHECK(rv.shape()[0] == 3);
  CHECK(rv.shape()[1] == 4);
  CHECK(rv.shape()[2] == 2);
  CHECK(rv.shape()[3] == 2);
  bool equal = true;
  for (std::size_t a = 0; a < 3; ++a) {
    for (std::size_t b = 0; b < 4; ++b) {
      for (std::size_t c = 0; c < 2; ++c) {
        for (std::size_t k = 0; k < 2; ++k) {
          equal = equal && (rv.control_points()[((a * 4 + b) * 2 + c) * 2 + k] ==
                            v.control_points()[((c * 3 + a) * 4 + b) * 2 + k]);
        }
      }
    }
  }
  CHECK(equal);
  // The identity permutation, and a curve.
  const std::vector<long long> ident = {0, 1, 2};
  CHECK(same_control_points<T>(pb::permuted<T>(v, std::span<const long long>(ident)), v));
  const pb::Bezier<T> c1 = random_bezier<T>(rng, {3}, 1, false);
  const std::vector<long long> zero = {0};
  CHECK(same_control_points<T>(pb::permuted<T>(c1, std::span<const long long>(zero)), c1));
}

TEST_CASE_TEMPLATE("bezier transformed: the image moves by the affine map", T, float, double) {
  Lcg rng;
  const std::vector<double> mat = {0.6, -0.8, 0.8, 0.6};
  const std::vector<double> off = {1.0, -2.0};
  const std::array<std::size_t, 2> mshape = {2, 2};
  const double rowsum = 1.4;  // the largest sum of |A_ij| over a row
  const double bmax = 2.0;
  for (const Case& cs : cases()) {
    const pb::Bezier<T> f = random_bezier<T>(rng, cs.degrees, 2, cs.rational);
    const pb::Bezier<T> g = pb::transformed<T>(f, mat, mshape, off);
    CHECK(g.rational() == cs.rational);
    CHECK(g.size() == f.size());
    const std::size_t dim = f.dim();
    const std::vector<T> pts = grid<T>(dim, dim == 1 ? samples() : coarse_samples());
    const NdArray<T> fp = eval_at(f, pts);
    const NdArray<T> gp = eval_at(g, pts);
    // The image of f under x -> A x + b, in double from the evaluated points.
    NdArray<T> want(std::vector<std::size_t>{fp.shape()[0], 2}, pantr::for_overwrite);
    for (std::size_t j = 0; j < fp.shape()[0]; ++j) {
      const double x = static_cast<double>(fp.data()[2 * j]);
      const double y = static_cast<double>(fp.data()[2 * j + 1]);
      want.data()[2 * j] = static_cast<T>(mat[0] * x + mat[1] * y + off[0]);
      want.data()[2 * j + 1] = static_cast<T>(mat[2] * x + mat[3] * y + off[1]);
    }
    // M' bounds the transformed control points: |A x| <= rowsum M plus |b| (times the weight,
    // at most 2, when rational). Rounding A and b to T, two products and two sums: (n + 3) eps M'
    // with n = 2; evaluating g: 8 S eps M'; evaluating f and mapping it through A: 8 S eps M
    // rowsum <= 8 S eps M'; the rounding of the reference to T: eps M'.
    const double m_f = max_abs<T>(f.control_points());
    const double m_g = rowsum * m_f + bmax * (cs.rational ? 2.0 : 1.0);
    const double tol =
        eps_of<T>() * m_g * (cs.rational ? 4.0 : 1.0) * (16.0 * sum_p1(f) + 2.0 + 3.0 + 1.0);
    const double err = max_diff(gp, want);
    CAPTURE(dim);
    CHECK(err <= tol);
    if (cs.rational) {
      // The weight column is copied unchanged.
      bool weights_equal = true;
      for (std::size_t r = 0; r < f.size() / f.cp_size(); ++r) {
        weights_equal =
            weights_equal && (g.control_points()[r * 3 + 2] == f.control_points()[r * 3 + 2]);
      }
      CHECK(weights_equal);
    }
  }
}

TEST_CASE_TEMPLATE("bezier transformed: exact on dyadic data under a power-of-two matrix", T, float,
                   double) {
  // A = 2 I, b = (1, 0.5) on control points that are multiples of 1/8: A x + b is exact. For a
  // rational Bezier the stored (w x, w) goes to (2 w x + w b, w).
  const std::vector<double> mat = {2.0, 0.0, 0.0, 2.0};
  const std::vector<double> off = {1.0, 0.5};
  const std::array<std::size_t, 2> mshape = {2, 2};
  const pb::Bezier<T> plain = make_bezier<T>(cast_all<T>({0.125, -0.5, 0.75, 0.25}), {2, 2});
  const pb::Bezier<T> moved = pb::transformed<T>(plain, mat, mshape, off);
  const std::array<double, 4> want = {1.25, -0.5, 2.5, 1.0};
  bool equal = true;
  for (std::size_t i = 0; i < 4; ++i) {
    equal = equal && (moved.control_points()[i] == static_cast<T>(want[i]));
  }
  CHECK(equal);
  // Rational, weights 2 and 0.5: (w x, y w, w) = (0.5, 1, 2) and (0.25, -0.125, 0.5).
  const pb::Bezier<T> rat =
      make_bezier<T>(cast_all<T>({0.5, 1.0, 2.0, 0.25, -0.125, 0.5}), {2, 3}, true);
  const pb::Bezier<T> rmoved = pb::transformed<T>(rat, mat, mshape, off);
  // 2 * 0.5 + 2 * 1 = 3, 2 * 1 + 2 * 0.5 = 3, weight 2; 2 * 0.25 + 0.5 = 1, 2 * -0.125 + 0.25 =
  // 0, weight 0.5.
  const std::array<double, 6> rwant = {3.0, 3.0, 2.0, 1.0, 0.0, 0.5};
  bool requal = true;
  for (std::size_t i = 0; i < 6; ++i) {
    requal = requal && (rmoved.control_points()[i] == static_cast<T>(rwant[i]));
  }
  CHECK(requal);
  CHECK(rmoved.rational());
}

TEST_CASE_TEMPLATE("bezier split, slice, boundary: the checks and their messages", T, float,
                   double) {
  Lcg rng;
  const pb::Bezier<T> surf = random_bezier<T>(rng, {2, 2}, 1, false);
  const pb::Bezier<T> curve = random_bezier<T>(rng, {3}, 1, false);
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::split<T>(surf, 2, 0.5)),
                       "direction must be in [0, 2), got 2.", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::split<T>(surf, -1, 0.5)),
                       "direction must be in [0, 2), got -1.", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::split<T>(surf, 0, 0.0)),
                       "value must be strictly inside (0, 1), got 0.0.", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::split<T>(surf, 0, 1.0)),
                       "value must be strictly inside (0, 1), got 1.0.", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::split<T>(surf, 1, -0.5)),
                       "value must be strictly inside (0, 1), got -0.5.", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::split<T>(surf, 1, -1e-5)),
                       "value must be strictly inside (0, 1), got -1e-05.", pantr::value_error);
  // The direction is checked before the value.
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::split<T>(surf, 3, 2.0)),
                       "direction must be in [0, 2), got 3.", pantr::value_error);

  CHECK_THROWS_WITH_AS(static_cast<void>(pb::slice<T>(surf, 2, 0.5)),
                       "axis must be in [0, 2), got 2.", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::slice<T>(surf, -1, 0.5)),
                       "axis must be in [0, 2), got -1.", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::slice<T>(surf, 0, 1.5)),
                       "value must be in [0, 1], got 1.5.", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::slice<T>(surf, 0, -0.25)),
                       "value must be in [0, 1], got -0.25.", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::slice<T>(surf, 5, 3.0)),
                       "axis must be in [0, 2), got 5.", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::slice_point<T>(curve, 1, 0.5)),
                       "axis must be in [0, 1), got 1.", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::slice_point<T>(curve, 0, 2.0)),
                       "value must be in [0, 1], got 2.0.", pantr::value_error);
  // One function per return type.
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::slice<T>(curve, 0, 0.5)),
                       "A 1D Bézier slices to a point, not a Bézier: call slice_point.",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::slice_point<T>(surf, 0, 0.5)),
                       "A Bézier of dimension 2 slices to a Bézier, not a point: call slice.",
                       pantr::value_error);

  CHECK_THROWS_WITH_AS(static_cast<void>(pb::boundary<T>(surf, 0, 2)),
                       "side must be 0 or 1, got 2.", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::boundary<T>(surf, 0, -1)),
                       "side must be 0 or 1, got -1.", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::boundary<T>(surf, 2, 0)),
                       "axis must be in [0, 2), got 2.", pantr::value_error);
  // The side is checked before the axis.
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::boundary<T>(surf, 7, 3)),
                       "side must be 0 or 1, got 3.", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::boundary_point<T>(curve, 0, 2)),
                       "side must be 0 or 1, got 2.", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::boundary_point<T>(curve, 1, 0)),
                       "axis must be in [0, 1), got 1.", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::boundary<T>(curve, 0, 0)),
                       "The boundary of a 1D Bézier is a point, not a Bézier: call "
                       "boundary_point.",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::boundary_point<T>(surf, 0, 0)),
                       "A Bézier of dimension 2 has a Bézier as boundary, not a point: call "
                       "boundary.",
                       pantr::value_error);
}

TEST_CASE_TEMPLATE("bezier restrict and collapse: the checks and their messages", T, float,
                   double) {
  Lcg rng;
  const pb::Bezier<T> surf = random_bezier<T>(rng, {2, 2}, 1, false);
  const pb::Bezier<T> curve = random_bezier<T>(rng, {3}, 1, false);
  const pb::Bezier<T> vol = random_bezier<T>(rng, {1, 1, 1}, 1, false);
  auto run = [&](const pb::Bezier<T>& b, const std::vector<Bounds>& bounds) {
    return pb::restrict<T>(b, std::span<const Bounds>(bounds));
  };
  CHECK_THROWS_WITH_AS(static_cast<void>(run(surf, {bd(0.1, 0.5)})),
                       "bounds sequence length (1) must match dim (2).", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(run(curve, {bd(0.1, 0.5), bd(0.1, 0.5)})),
                       "bounds sequence length (2) must match dim (1).", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(run(surf, {bd(0.1, 0.5), bd(0.5, 0.5)})),
                       "Lower bound (0.5) must be strictly less than upper bound (0.5) in "
                       "direction 1.",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(run(curve, {bd(0.7, 0.2)})),
                       "Lower bound (0.7) must be strictly less than upper bound (0.2) in "
                       "direction 0.",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(run(surf, {bd(0.2, 1.5), bd(0.1, 0.5)})),
                       "Bounds (0.2, 1.5) must lie within [0, 1] in direction 0.",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(run(surf, {std::nullopt, bd(-0.1, 0.5)})),
                       "Bounds (-0.1, 0.5) must lie within [0, 1] in direction 1.",
                       pantr::value_error);
  // Direction 0 is checked completely before direction 1; within a direction, lower >= upper
  // comes before the range.
  CHECK_THROWS_WITH_AS(static_cast<void>(run(surf, {bd(0.2, 1.5), bd(0.7, 0.2)})),
                       "Bounds (0.2, 1.5) must lie within [0, 1] in direction 0.",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(run(curve, {bd(1.5, 0.2)})),
                       "Lower bound (1.5) must be strictly less than upper bound (0.2) in "
                       "direction 0.",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(run(curve, {bd(0.0, 1.0)})),
                       "Bounds match the full domain; at least one direction must be restricted.",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(run(surf, {std::nullopt, std::nullopt})),
                       "Bounds match the full domain; at least one direction must be restricted.",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(run(surf, {bd(0.0, 1.0), std::nullopt})),
                       "Bounds match the full domain; at least one direction must be restricted.",
                       pantr::value_error);

  const std::vector<T> one = cast_all<T>({0.5});
  const std::vector<T> two = cast_all<T>({0.5, 0.5});
  const std::vector<T> three = cast_all<T>({0.5, 0.5, 0.5});
  const std::vector<T> bad = cast_all<T>({0.5, 1.5});
  const std::vector<T> neg = cast_all<T>({-0.25, 0.5});
  CHECK_THROWS_WITH_AS(
      static_cast<void>(pb::collapse_along_axis<T>(curve, 0, std::span<const T>(one))),
      "collapse_along_axis requires dim >= 2.", pantr::value_error);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(pb::collapse_along_axis<T>(surf, 2, std::span<const T>(one))),
      "axis must be in [0, 2), got 2.", pantr::value_error);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(pb::collapse_along_axis<T>(surf, -1, std::span<const T>(one))),
      "axis must be in [0, 2), got -1.", pantr::value_error);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(pb::collapse_along_axis<T>(surf, 0, std::span<const T>(two))),
      "values must have length dim - 1 = 1, got shape (2,).", pantr::value_error);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(pb::collapse_along_axis<T>(vol, 1, std::span<const T>(three))),
      "values must have length dim - 1 = 2, got shape (3,).", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::collapse_along_axis<T>(vol, 1, std::span<const T>())),
                       "values must have length dim - 1 = 2, got shape (0,).", pantr::value_error);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(pb::collapse_along_axis<T>(vol, 1, std::span<const T>(bad))),
      "All values must be in [0, 1].", pantr::value_error);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(pb::collapse_along_axis<T>(vol, 0, std::span<const T>(neg))),
      "All values must be in [0, 1].", pantr::value_error);
  // The length is checked before the range, and the axis before the length.
  CHECK_THROWS_WITH_AS(
      static_cast<void>(pb::collapse_along_axis<T>(surf, 0, std::span<const T>(bad))),
      "values must have length dim - 1 = 1, got shape (2,).", pantr::value_error);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(pb::collapse_along_axis<T>(surf, 4, std::span<const T>(bad))),
      "axis must be in [0, 2), got 4.", pantr::value_error);
}

TEST_CASE_TEMPLATE("bezier reversed, permuted, transformed: the checks and their messages", T,
                   float, double) {
  Lcg rng;
  const pb::Bezier<T> surf = random_bezier<T>(rng, {2, 1}, 2, false);
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::reversed<T>(surf, 2)),
                       "direction must be in [0, 2), got 2.", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::reversed<T>(surf, -1)),
                       "direction must be in [0, 2), got -1.", pantr::value_error);

  const std::string msg = "permutation must be a permutation of range(2)";
  const std::vector<long long> too_short = {0};
  const std::vector<long long> too_long = {0, 1, 0};
  const std::vector<long long> repeated = {1, 1};
  const std::vector<long long> out_of_range = {0, 2};
  const std::vector<long long> negative = {0, -1};
  CHECK_THROWS_WITH_AS(
      static_cast<void>(pb::permuted<T>(surf, std::span<const long long>(too_short))), msg.c_str(),
      pantr::value_error);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(pb::permuted<T>(surf, std::span<const long long>(too_long))), msg.c_str(),
      pantr::value_error);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(pb::permuted<T>(surf, std::span<const long long>(repeated))), msg.c_str(),
      pantr::value_error);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(pb::permuted<T>(surf, std::span<const long long>(out_of_range))),
      msg.c_str(), pantr::value_error);
  CHECK_THROWS_WITH_AS(
      static_cast<void>(pb::permuted<T>(surf, std::span<const long long>(negative))), msg.c_str(),
      pantr::value_error);

  const std::vector<double> m3 = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const std::vector<double> m2 = {1, 0, 0, 1};
  const std::vector<double> m23 = {1, 0, 0, 0, 1, 0};
  const std::vector<double> off2 = {0.0, 0.0};
  const std::array<std::size_t, 2> s33 = {3, 3};
  const std::array<std::size_t, 2> s22 = {2, 2};
  const std::array<std::size_t, 2> s23 = {2, 3};
  const std::array<std::size_t, 1> s4 = {4};
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::transformed<T>(surf, m3, s33, off2)),
                       "Transform dimension (3) does not match the geometric rank (2) of the "
                       "control points.",
                       pantr::value_error);
  // The geometric rank excludes the weight: a rational rank-2 Bezier has three coordinates.
  const pb::Bezier<T> rat = random_bezier<T>(rng, {1}, 2, true);
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::transformed<T>(rat, m3, s33, off2)),
                       "Transform dimension (3) does not match the geometric rank (2) of the "
                       "control points.",
                       pantr::value_error);
  // A matrix of shape (2, 3) is refused with the first extent in the message, as in Python.
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::transformed<T>(surf, m23, s23, off2)),
                       "Transform dimension (2) does not match the geometric rank (2) of the "
                       "control points.",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::transformed<T>(surf, m2, s4, off2)),
                       "matrix must be a 2D array, got shape (4,).", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::transformed<T>(surf, m23, s22, off2)),
                       "matrix holds 6 values, but the shape (2, 2) needs 4.", pantr::value_error);
  const std::vector<double> off3 = {0.0, 0.0, 0.0};
  CHECK_THROWS_WITH_AS(static_cast<void>(pb::transformed<T>(surf, m2, s22, off3)),
                       "offset must have length 2, got 3.", pantr::value_error);
}
