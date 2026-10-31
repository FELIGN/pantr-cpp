// SpanwiseElementExtraction and the Kronecker application behind it (`apply_kron`).
//
// Oracle. The element operator is `M = kron(M_0, ..., M_{d-1})` of the per-direction operators.
// `M v`, `M^T v`, `M^T K M` and `M K M^T` are compared with the dense product of the exact
// rational values of the stored operands, formed by the definition of the Kronecker product. The
// library contracts one direction at a time (and one side at a time), each a recursive sum of `n`
// products, so an entry of the result is off by at most `gamma(Sigma n) (|M| |v|)` (Higham,
// Accuracy and Stability of Numerical Algorithms, 2nd ed., section 3.5, applied to the product
// `A_s ... A_1 v` whose factors `A_k = I x M_k x I` have `|A_s| ... |A_1| = |M|`): `Sigma n` is the
// sum of the contracted lengths of all stages, `gamma(n) = n u / (1 - n u)`. The comparison is
// against that bound with no safety factor; rounding of the bound itself is a relative `1e-9`.
// The per-direction operators of the class are compared with the 1D operator functions of
// `extraction.hpp` (tested in test_bspline_extraction.cpp), not with a recomputation of the class.
#include <doctest/doctest.h>

#include "bspline_oracle.hpp"
#include "bspline_test_support.hpp"

#include <pantr/basis/lagrange_variant.hpp>
#include <pantr/bspline/detail/kron_apply.hpp>
#include <pantr/bspline/extraction.hpp>
#include <pantr/bspline/space.hpp>
#include <pantr/bspline/spanwise_extraction.hpp>
#include <pantr/core/errors.hpp>
#include <pantr/core/ndarray.hpp>
#include <pantr/parallel/pool.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace bs = pantr::bspline;
namespace oracle = bspline_oracle;
using bspline_test::Dir;
using bspline_test::message_of;
using oracle::Rational;
using pantr::basis::LagrangeVariant;

namespace {

/// A dense matrix with its entrywise absolute values (in double, for the error bound) and,
/// unless `exact` is false, its exact values. A matrix without exact values is only a bound.
struct Dense {
  std::size_t rows = 0;
  std::size_t cols = 0;
  bool exact = true;
  std::vector<Rational> v;
  std::vector<double> a;
};

Dense make_dense(std::size_t rows, std::size_t cols, bool exact) {
  return Dense{rows, cols, exact, std::vector<Rational>(exact ? rows * cols : 0),
               std::vector<double>(rows * cols)};
}

/// The dense matrix of stored values.
template <class T>
Dense dense_of(std::span<const T> p, std::size_t rows, std::size_t cols, bool exact = true) {
  Dense m = make_dense(rows, cols, exact);
  for (std::size_t i = 0; i < rows * cols; ++i) {
    if (exact) {
      m.v[i] = oracle::exact(static_cast<double>(p[i]));
    }
    m.a[i] = std::abs(static_cast<double>(p[i]));
  }
  return m;
}

Dense identity_dense(std::size_t n, bool exact = true) {
  Dense m = make_dense(n, n, exact);
  for (std::size_t i = 0; i < n; ++i) {
    if (exact) {
      m.v[i * n + i] = 1;
    }
    m.a[i * n + i] = 1.0;
  }
  return m;
}

Dense transpose(const Dense& m) {
  Dense t = make_dense(m.cols, m.rows, m.exact);
  for (std::size_t i = 0; i < m.rows; ++i) {
    for (std::size_t j = 0; j < m.cols; ++j) {
      if (m.exact) {
        t.v[j * m.rows + i] = m.v[i * m.cols + j];
      }
      t.a[j * m.rows + i] = m.a[i * m.cols + j];
    }
  }
  return t;
}

Dense multiply(const Dense& x, const Dense& y) {
  REQUIRE(x.cols == y.rows);
  REQUIRE(x.exact == y.exact);
  Dense r = make_dense(x.rows, y.cols, x.exact);
  for (std::size_t i = 0; i < x.rows; ++i) {
    for (std::size_t k = 0; k < x.cols; ++k) {
      for (std::size_t j = 0; j < y.cols; ++j) {
        if (x.exact) {
          r.v[i * y.cols + j] += x.v[i * x.cols + k] * y.v[k * y.cols + j];
        }
        r.a[i * y.cols + j] += x.a[i * x.cols + k] * y.a[k * y.cols + j];
      }
    }
  }
  return r;
}

/// `kron(a, b)`: entry `(i * rb + p, j * cb + q)` is `a(i, j) b(p, q)`.
Dense kron2(const Dense& x, const Dense& y) {
  REQUIRE(x.exact == y.exact);
  Dense r = make_dense(x.rows * y.rows, x.cols * y.cols, x.exact);
  for (std::size_t i = 0; i < x.rows; ++i) {
    for (std::size_t j = 0; j < x.cols; ++j) {
      for (std::size_t p = 0; p < y.rows; ++p) {
        for (std::size_t q = 0; q < y.cols; ++q) {
          const std::size_t at = (i * y.rows + p) * r.cols + j * y.cols + q;
          if (x.exact) {
            r.v[at] = x.v[i * x.cols + j] * y.v[p * y.cols + q];
          }
          r.a[at] = x.a[i * x.cols + j] * y.a[p * y.cols + q];
        }
      }
    }
  }
  return r;
}

Dense kron_all(const std::vector<Dense>& factors) {
  Dense m = factors[0];
  for (std::size_t k = 1; k < factors.size(); ++k) {
    m = kron2(m, factors[k]);
  }
  return m;
}

template <class T>
double unit_roundoff() {
  return 0.5 * static_cast<double>(std::numeric_limits<T>::epsilon());
}

double gamma_of(std::size_t n, double u) {
  const double nu = static_cast<double>(n) * u;
  return nu / (1.0 - nu);
}

/// Deterministic values in [-1, 1) with a full mantissa (not dyadic), stored as `T`.
template <class T>
std::vector<T> random_values(std::size_t count, unsigned seed) {
  std::vector<T> v(count);
  std::uint64_t s = 0x9E3779B97F4A7C15ULL + seed * 0xBF58476D1CE4E5B9ULL;
  for (T& x : v) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    x = static_cast<T>(static_cast<double>(s >> 11) / 4503599627370496.0 - 1.0);
  }
  return v;
}

/// The result `got` is within the bound of the exact `want`, entry by entry. `terms` is the sum of
/// the contracted lengths of the stages that formed it.
template <class T>
void expect_within(std::span<const T> got, const Dense& want, std::size_t terms) {
  REQUIRE(got.size() == want.v.size());
  const double g = gamma_of(terms, unit_roundoff<T>());
  bool all_ok = true;
  std::size_t first_bad = 0;
  for (std::size_t i = 0; i < got.size() && all_ok; ++i) {
    const double bound = g * want.a[i] * (1.0 + 1e-9);
    all_ok = oracle::within(static_cast<double>(got[i]), want.v[i], bound);
    first_bad = i;
  }
  CHECK_MESSAGE(all_ok, "entry " << first_bad << " of " << got.size() << " is off by more than "
                                 << g << " times the sum of the terms' magnitudes");
}

/// `got` and `other`, two results of one computation, agree within twice the bound each has
/// against the exact value (`want.a` holds the magnitudes of the terms).
template <class T>
void expect_agree(std::span<const T> got, std::span<const T> other, const Dense& want,
                  std::size_t terms) {
  REQUIRE(got.size() == other.size());
  REQUIRE(got.size() == want.a.size());
  const double g = gamma_of(terms, unit_roundoff<T>());
  bool all_ok = true;
  std::size_t first_bad = 0;
  for (std::size_t i = 0; i < got.size() && all_ok; ++i) {
    const double bound = 2.0 * g * want.a[i] * (1.0 + 1e-9);
    all_ok = std::abs(static_cast<double>(got[i]) - static_cast<double>(other[i])) <= bound;
    first_bad = i;
  }
  CHECK_MESSAGE(all_ok, "entry " << first_bad << " of " << got.size()
                                 << " differs by more than twice the derived bound");
}

/// One direction's factor shape and flag.
struct Shape {
  std::size_t rows;
  std::size_t cols;
  bool identity;
};

/// The reference of an operation on the Kronecker product of `factors`.
struct Reference {
  Dense result;
  std::size_t terms;  // the sum of the contracted lengths
};

/// The exact result of `op` on `operand` (a column vector `(n, 1)` or a square matrix), and the
/// number of terms of its stages.
Reference reference_of(bs::ExtractionOp op, const std::vector<Dense>& factors,
                       const Dense& operand) {
  const Dense m = kron_all(factors);
  std::size_t from_sum = 0;
  const bool transposed = op == bs::ExtractionOp::apply_transpose || op == bs::ExtractionOp::mt_k_m;
  for (const Dense& f : factors) {
    from_sum += transposed ? f.rows : f.cols;
  }
  switch (op) {
    case bs::ExtractionOp::apply:
      return {multiply(m, operand), from_sum};
    case bs::ExtractionOp::apply_transpose:
      return {multiply(transpose(m), operand), from_sum};
    case bs::ExtractionOp::mt_k_m:
      return {multiply(transpose(m), multiply(operand, m)), 2 * from_sum};
    case bs::ExtractionOp::m_k_mt:
      return {multiply(m, multiply(operand, transpose(m))), 2 * from_sum};
  }
  return {};
}

constexpr bs::ExtractionOp kAllOps[] = {bs::ExtractionOp::apply, bs::ExtractionOp::apply_transpose,
                                        bs::ExtractionOp::mt_k_m, bs::ExtractionOp::m_k_mt};

bool is_vector_op(bs::ExtractionOp op) {
  return op == bs::ExtractionOp::apply || op == bs::ExtractionOp::apply_transpose;
}

/// The operand extents an operation reads: `prod(from)` entries (a vector), or that square.
std::size_t operand_side(bs::ExtractionOp op, const std::vector<Shape>& shapes) {
  const bool transposed = op == bs::ExtractionOp::apply_transpose || op == bs::ExtractionOp::mt_k_m;
  std::size_t n = 1;
  for (const Shape& s : shapes) {
    n *= transposed ? s.rows : s.cols;
  }
  return n;
}

std::size_t result_side(bs::ExtractionOp op, const std::vector<Shape>& shapes) {
  const bool transposed = op == bs::ExtractionOp::apply_transpose || op == bs::ExtractionOp::mt_k_m;
  std::size_t n = 1;
  for (const Shape& s : shapes) {
    n *= transposed ? s.cols : s.rows;
  }
  return n;
}

/// Run `op` through the raw `apply_kron` on factor data and compare it with the reference.
template <class T>
void check_apply_kron(bs::ExtractionOp op, const std::vector<Shape>& shapes, unsigned seed) {
  std::vector<std::vector<T>> data;
  std::vector<bs::KronFactor<T>> factors;
  std::vector<Dense> dense;
  std::vector<std::size_t> rows;
  std::vector<std::size_t> cols;
  for (std::size_t k = 0; k < shapes.size(); ++k) {
    const Shape& s = shapes[k];
    data.push_back(random_values<T>(s.rows * s.cols, seed + static_cast<unsigned>(k)));
    rows.push_back(s.rows);
    cols.push_back(s.cols);
  }
  for (std::size_t k = 0; k < shapes.size(); ++k) {
    const Shape& s = shapes[k];
    if (s.identity) {
      // An identity factor is not read: hand it garbage, and let the reference use the identity.
      std::fill(data[k].begin(), data[k].end(), static_cast<T>(7));
      dense.push_back(identity_dense(s.rows));
    } else {
      dense.push_back(dense_of<T>(data[k], s.rows, s.cols));
    }
    factors.push_back({std::span<const T>(data[k]), s.rows, s.cols, s.identity});
  }
  const std::size_t n_op = operand_side(op, shapes);
  const std::size_t n_res = result_side(op, shapes);
  const bool vec = is_vector_op(op);
  const std::vector<T> operand = random_values<T>(vec ? n_op : n_op * n_op, seed + 100);
  const Dense operand_dense = dense_of<T>(operand, vec ? n_op : n_op, vec ? 1 : n_op);
  const Reference ref = reference_of(op, dense, operand_dense);
  std::vector<T> out(vec ? n_res : n_res * n_res, static_cast<T>(-3));
  // The scratch is exactly the size asked for, so that an overrun is a heap overflow the
  // sanitizer sees.
  std::vector<T> scratch(bs::required_scratch_size(op, rows, cols));
  bs::apply_kron<T>(op, factors, operand, out, scratch);
  expect_within<T>(out, ref.result, ref.terms);
}

/// The identity patterns tried in dimension `d`: all of them up to 3, five of them for 4.
std::vector<std::vector<bool>> patterns(std::size_t d) {
  std::vector<std::vector<bool>> out;
  if (d == 4) {
    return {{false, false, false, false},
            {true, true, true, true},
            {true, false, true, false},
            {false, true, true, false},
            {false, false, false, true}};
  }
  for (std::size_t bits = 0; bits < (std::size_t{1} << d); ++bits) {
    std::vector<bool> p(d);
    for (std::size_t k = 0; k < d; ++k) {
      p[k] = ((bits >> k) & 1U) != 0;
    }
    out.push_back(p);
  }
  return out;
}

std::vector<Shape> shapes_for(const std::vector<bool>& identity) {
  std::vector<Shape> shapes;
  for (std::size_t k = 0; k < identity.size(); ++k) {
    // Rectangular where not identity: alternately 2 x 3 and 3 x 2.
    const std::size_t rows = 2 + (k % 2);
    const std::size_t cols = 3 - (k % 2);
    shapes.push_back(identity[k] ? Shape{rows, rows, true} : Shape{rows, cols, false});
  }
  return shapes;
}

}  // namespace

TEST_CASE_TEMPLATE("apply_kron: the four operations equal the dense Kronecker product", T, double,
                   float) {
  unsigned seed = 11;
  for (std::size_t d = 1; d <= 4; ++d) {
    for (const std::vector<bool>& identity : patterns(d)) {
      const std::vector<Shape> shapes = shapes_for(identity);
      for (const bs::ExtractionOp op : kAllOps) {
        CAPTURE(d);
        CAPTURE(static_cast<int>(op));
        CAPTURE(identity);
        check_apply_kron<T>(op, shapes, seed);
        seed += 7;
      }
    }
  }
}

TEST_CASE_TEMPLATE("apply_kron: skipping an identity factor equals multiplying by it", T, double,
                   float) {
  // The same factors, one direction flagged identity and given as an explicit identity matrix: the
  // skipped stage is a copy, the explicit one multiplies by 1 and adds exact zeros, so the two
  // agree to the last bit on finite data.
  const std::vector<T> a = random_values<T>(6, 3);
  const std::vector<T> eye = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  const std::vector<T> c = random_values<T>(4, 5);
  const std::vector<T> garbage(9, static_cast<T>(99));
  for (const bs::ExtractionOp op : kAllOps) {
    const std::vector<bs::KronFactor<T>> skipped = {
        {a, 2, 3, false}, {garbage, 3, 3, true}, {c, 2, 2, false}};
    const std::vector<bs::KronFactor<T>> explicit_eye = {
        {a, 2, 3, false}, {eye, 3, 3, false}, {c, 2, 2, false}};
    const std::vector<std::size_t> rows = {2, 3, 2};
    const std::vector<std::size_t> cols = {3, 3, 2};
    const std::size_t n_in = operand_side(op, {{2, 3, false}, {3, 3, false}, {2, 2, false}});
    const std::size_t n_out = result_side(op, {{2, 3, false}, {3, 3, false}, {2, 2, false}});
    const bool vec = is_vector_op(op);
    const std::vector<T> operand = random_values<T>(vec ? n_in : n_in * n_in, 9);
    std::vector<T> out_skip(vec ? n_out : n_out * n_out);
    std::vector<T> out_eye(out_skip.size());
    std::vector<T> scratch(bs::required_scratch_size(op, rows, cols));
    bs::apply_kron<T>(op, skipped, operand, out_skip, scratch);
    bs::apply_kron<T>(op, explicit_eye, operand, out_eye, scratch);
    CHECK(out_skip == out_eye);
  }
}

TEST_CASE_TEMPLATE("apply_kron: every direction identity copies the operand, in place too", T,
                   double, float) {
  const std::vector<std::size_t> rows = {2, 3};
  const std::vector<std::size_t> cols = {2, 3};
  const std::vector<T> junk(9, static_cast<T>(5));
  const std::vector<bs::KronFactor<T>> factors = {{junk, 2, 2, true}, {junk, 3, 3, true}};
  for (const bs::ExtractionOp op : kAllOps) {
    const bool vec = is_vector_op(op);
    const std::size_t n = vec ? 6 : 36;
    std::vector<T> operand = random_values<T>(n, 17);
    const std::vector<T> keep = operand;
    std::vector<T> out(n);
    // The scratch is sized for any pattern, an all-identity cell included.
    std::vector<T> scratch(bs::required_scratch_size(op, rows, cols));
    bs::apply_kron<T>(op, factors, operand, out, scratch);
    CHECK(out == keep);
    // In place: `out` is the operand.
    bs::apply_kron<T>(op, factors, operand, operand, scratch);
    CHECK(operand == keep);
    // Partly overlapping ranges are a copy too, and correct.
    std::vector<T> buf = random_values<T>(n + 2, 21);
    const std::vector<T> expected(buf.begin(), buf.begin() + static_cast<std::ptrdiff_t>(n));
    bs::apply_kron<T>(op, factors, std::span<const T>(buf.data(), n),
                      std::span<T>(buf.data() + 2, n), scratch);
    CHECK(std::vector<T>(buf.begin() + 2, buf.end()) == expected);
  }
}

TEST_CASE("apply_kron: refuses what cannot be applied, with the messages of the Python class") {
  const std::vector<double> a = {1, 2, 3, 4, 5, 6};
  const std::vector<double> sq = {1, 0, 0, 1};
  std::vector<double> out(6);
  std::vector<double> scratch(64);
  const std::vector<double> operand(3, 1.0);
  const std::vector<bs::KronFactor<double>> none;
  CHECK(message_of([&] {
          bs::apply_kron<double>(bs::ExtractionOp::apply, none, operand, out, scratch);
        }) == "At least one direction is required");
  const std::vector<bs::KronFactor<double>> bad_identity = {{a, 2, 3, true}};
  CHECK(message_of([&] {
          bs::apply_kron<double>(bs::ExtractionOp::apply, bad_identity, operand, out, scratch);
        }) ==
        "factors[0] is flagged identity but has shape (2, 3); an identity operator must be "
        "square");
  // One value too many is refused as one too few is: the count is `rows * cols` exactly.
  const std::vector<bs::KronFactor<double>> oversize = {{a, 2, 2, false}};
  CHECK(message_of([&] {
          bs::apply_kron<double>(bs::ExtractionOp::apply, oversize,
                                 std::span<const double>(operand).first(2),
                                 std::span<double>(out).first(2), scratch);
        }) == "factors[0] holds 6 values, but its shape needs 2 x 2");
  const std::vector<bs::KronFactor<double>> short_values = {{sq, 2, 3, false}};
  CHECK(message_of([&] {
          bs::apply_kron<double>(bs::ExtractionOp::apply, short_values, operand, out, scratch);
        }).find("factors[0] holds 4 values") == 0);
  const std::vector<bs::KronFactor<double>> ok = {{a, 2, 3, false}};
  CHECK(message_of([&] {
          bs::apply_kron<double>(bs::ExtractionOp::apply, ok,
                                 std::span<const double>(operand).first(2),
                                 std::span<double>(out).first(2), scratch);
        }).find("Operand has 2 values, but expected 3") == 0);
  CHECK(message_of([&] {
          bs::apply_kron<double>(bs::ExtractionOp::apply, ok, operand,
                                 std::span<double>(out).first(3), scratch);
        }).find("Output array has 3 values, but expected 2") == 0);
  // d = 2 needs scratch.
  const std::vector<bs::KronFactor<double>> two = {{a, 2, 3, false}, {a, 2, 3, false}};
  std::vector<double> v9(9, 1.0);
  std::vector<double> o4(4);
  std::vector<double> short_scratch(1);
  CHECK(message_of([&] {
          bs::apply_kron<double>(bs::ExtractionOp::apply, two, v9, o4, short_scratch);
        }) == "Scratch size 1 is smaller than required 12");
  CHECK(message_of([&] {
          bs::apply_kron<double>(bs::ExtractionOp::apply, two, std::span<const double>(v9), o4,
                                 std::span<double>(v9.data(), 12).first(0));
        }).find("Scratch size 0") == 0);
  // out overlapping the operand, and the scratch overlapping either.
  std::vector<double> buf(40, 1.0);
  CHECK(message_of([&] {
          bs::apply_kron<double>(
              bs::ExtractionOp::apply, two, std::span<const double>(buf.data(), 9),
              std::span<double>(buf.data() + 8, 4), std::span<double>(buf.data() + 20, 12));
        }) == "out must not alias the operand; the kernel reads and writes overlapping memory");
  CHECK(message_of([&] {
          bs::apply_kron<double>(
              bs::ExtractionOp::apply, two, std::span<const double>(buf.data(), 9),
              std::span<double>(buf.data() + 9, 4), std::span<double>(buf.data() + 5, 12));
        }) == "scratch must not overlap the operand or out");
  // Adjacent ranges are disjoint.
  CHECK(message_of([&] {
          bs::apply_kron<double>(
              bs::ExtractionOp::apply, two, std::span<const double>(buf.data(), 9),
              std::span<double>(buf.data() + 9, 4), std::span<double>(buf.data() + 13, 12));
        }) == "<no exception>");
  CHECK_THROWS_AS(static_cast<void>(bs::to_extraction_op(4)), pantr::value_error);
  CHECK_THROWS_AS(static_cast<void>(bs::to_extraction_op(-1)), pantr::value_error);
}

TEST_CASE("required_scratch_size: twice the larger intermediate result, for any identity pattern") {
  // Worked by hand. apply, d = 2, rows (5, 6), cols (3, 4): the only intermediate is
  // rows[0] * cols[1] = 20 entries, twice that is 40.
  CHECK(bs::required_scratch_size(bs::ExtractionOp::apply, std::vector<std::size_t>{5, 6},
                                  std::vector<std::size_t>{3, 4}) == 40);
  // apply_transpose reads the rows and writes the columns: intermediate cols[0] * rows[1] = 18.
  CHECK(bs::required_scratch_size(bs::ExtractionOp::apply_transpose, std::vector<std::size_t>{5, 6},
                                  std::vector<std::size_t>{3, 4}) == 36);
  // One direction: the one stage writes the result, nothing is staged.
  CHECK(bs::required_scratch_size(bs::ExtractionOp::apply, std::vector<std::size_t>{3},
                                  std::vector<std::size_t>{5}) == 0);
  // Two-sided, d = 1: the row stage's result is (cols, rows) = 3 * 5 entries for M^T K M with
  // rows 5, cols 3.
  CHECK(bs::required_scratch_size(bs::ExtractionOp::mt_k_m, std::vector<std::size_t>{5},
                                  std::vector<std::size_t>{3}) == 30);
  CHECK(bs::required_scratch_size(bs::ExtractionOp::m_k_mt, std::vector<std::size_t>{5},
                                  std::vector<std::size_t>{3}) == 30);
  // Two-sided, d = 2, rows (2, 3), cols (3, 2), M K M^T: K has the extents (3, 2, 3, 2) and the
  // stages give (2, 2, 3, 2) = 24, (2, 2, 2, 2) = 16, (2, 3, 2, 2) = 24; the fourth writes the
  // result (2, 3, 2, 3). The larger intermediate is 24, so 48.
  CHECK(bs::required_scratch_size(bs::ExtractionOp::m_k_mt, std::vector<std::size_t>{2, 3},
                                  std::vector<std::size_t>{3, 2}) == 48);
  CHECK_THROWS_AS(
      static_cast<void>(bs::required_scratch_size(
          bs::ExtractionOp::apply, std::vector<std::size_t>{2, 3}, std::vector<std::size_t>{2})),
      pantr::value_error);
  CHECK_THROWS_AS(
      static_cast<void>(bs::required_scratch_size(
          bs::ExtractionOp::apply, std::vector<std::size_t>{}, std::vector<std::size_t>{})),
      pantr::value_error);
  CHECK_THROWS_AS(
      static_cast<void>(bs::required_scratch_size(
          bs::ExtractionOp::apply, std::vector<std::size_t>{0}, std::vector<std::size_t>{2})),
      pantr::value_error);
}

// ---------------------------------------------------------------------------------------------
// The class on real spaces.

namespace {

constexpr bs::ExtractionTarget kAllTargets[] = {
    bs::ExtractionTarget::bezier, bs::ExtractionTarget::lagrange, bs::ExtractionTarget::cardinal};

/// A direction with a mix of identity and non-identity elements: the first element is a Bézier
/// patch (multiplicity `degree + 1` at both its ends), the next ones are smooth, the interior of
/// the later ones is uniform.
Dir mixed_dir(int degree, int uniform_elements) {
  std::vector<double> interior;
  const auto order = static_cast<std::size_t>(degree) + 1;
  for (std::size_t m = 0; m < order; ++m) {
    interior.push_back(0.25);  // a knot of multiplicity degree + 1: C^-1
  }
  for (int i = 1; i <= uniform_elements; ++i) {
    interior.push_back(0.25 + 0.75 * static_cast<double>(i) /
                                  (static_cast<double>(uniform_elements) + 1.0));
  }
  return Dir{bspline_test::clamped(degree, interior), degree, false};
}

template <class T>
bs::SpanwiseElementExtraction<T> make_extraction(
    const std::vector<Dir>& dirs, bs::ExtractionTarget target,
    LagrangeVariant variant = LagrangeVariant::equispaces) {
  return bs::SpanwiseElementExtraction<T>(bspline_test::space_of<T>(dirs), target, variant);
}

/// The 1D operators `(n_elements, order, order)` of one direction by the extraction functions.
template <class T>
pantr::NdArray<T> operators_of(const bs::BsplineSpace1D<T>& s, bs::ExtractionTarget target,
                               LagrangeVariant variant) {
  switch (target) {
    case bs::ExtractionTarget::bezier:
      return bs::bezier_extraction_operators<T>(s);
    case bs::ExtractionTarget::lagrange: {
      pantr::NdArray<T> a({static_cast<std::size_t>(s.num_intervals()),
                           static_cast<std::size_t>(s.degree()) + 1,
                           static_cast<std::size_t>(s.degree()) + 1});
      bs::lagrange_extraction_operators<T>(s, variant, a.template view<3>());
      return a;
    }
    case bs::ExtractionTarget::cardinal: {
      pantr::NdArray<T> a({static_cast<std::size_t>(s.num_intervals()),
                           static_cast<std::size_t>(s.degree()) + 1,
                           static_cast<std::size_t>(s.degree()) + 1});
      bs::cardinal_extraction_operators<T>(s, a.template view<3>());
      return a;
    }
  }
  return pantr::NdArray<T>({1});
}

/// Everything a reference needs about a cell: the exact factors.
template <class T>
std::vector<Dense> factors_of_cell(const bs::BsplineSpace<T>& space,
                                   const std::vector<pantr::NdArray<T>>& full,
                                   const std::vector<std::size_t>& cell) {
  std::vector<Dense> factors;
  for (std::size_t k = 0; k < space.dim(); ++k) {
    const auto order = static_cast<std::size_t>(space.spaces()[k].degree()) + 1;
    factors.push_back(dense_of<T>(
        std::span<const T>(full[k].data() + cell[k] * order * order, order * order), order, order));
  }
  return factors;
}

/// The integer square root of a perfect square.
std::size_t isqrt(std::size_t n) {
  std::size_t r = 0;
  while (r * r < n) {
    ++r;
  }
  REQUIRE(r * r == n);
  return r;
}

std::vector<std::size_t> unravel(const std::vector<std::size_t>& extents, std::size_t flat) {
  std::vector<std::size_t> cell(extents.size());
  for (std::size_t k = extents.size(); k-- > 0;) {
    cell[k] = flat % extents[k];
    flat /= extents[k];
  }
  return cell;
}

/// Run one operation of the class on one cell into a fresh vector.
template <class T>
std::vector<T> run_cell(const bs::SpanwiseElementExtraction<T>& e, bs::ExtractionOp op,
                        const std::vector<T>& operand, const std::vector<std::size_t>& cell) {
  std::size_t n_in = 1;
  std::size_t n_out = 1;
  for (std::size_t k = 0; k < e.dim(); ++k) {
    n_in *= e.input_shape_per_dir()[k];
    n_out *= e.output_shape_per_dir()[k];
  }
  switch (op) {
    case bs::ExtractionOp::apply: {
      std::vector<T> out(n_out);
      e.apply(operand, cell, out);
      return out;
    }
    case bs::ExtractionOp::apply_transpose: {
      std::vector<T> out(n_in);
      e.apply_transpose(operand, cell, out);
      return out;
    }
    case bs::ExtractionOp::mt_k_m: {
      std::vector<T> out(n_in * n_in);
      e.apply_mt_k_m(pantr::mdview<const T, 2>(operand.data(), n_out, n_out), cell,
                     pantr::mdview<T, 2>(out.data(), n_in, n_in));
      return out;
    }
    case bs::ExtractionOp::m_k_mt: {
      std::vector<T> out(n_out * n_out);
      e.apply_m_k_mt(pantr::mdview<const T, 2>(operand.data(), n_in, n_in), cell,
                     pantr::mdview<T, 2>(out.data(), n_out, n_out));
      return out;
    }
  }
  return {};
}

template <class T>
std::size_t operand_entries(const bs::SpanwiseElementExtraction<T>& e, bs::ExtractionOp op) {
  std::size_t n_in = 1;
  std::size_t n_out = 1;
  for (std::size_t k = 0; k < e.dim(); ++k) {
    n_in *= e.input_shape_per_dir()[k];
    n_out *= e.output_shape_per_dir()[k];
  }
  switch (op) {
    case bs::ExtractionOp::apply:
      return n_in;
    case bs::ExtractionOp::apply_transpose:
      return n_out;
    case bs::ExtractionOp::mt_k_m:
      return n_out * n_out;
    case bs::ExtractionOp::m_k_mt:
      return n_in * n_in;
  }
  return 0;
}

/// The directions of the dimension-`d` test spaces: degrees small enough that the exact reference
/// stays cheap in dimension 4.
std::vector<Dir> dirs_for(std::size_t d) {
  const int degrees[] = {2, 3, 1, 2};
  const int uniform[] = {3, 2, 3, 2};
  std::vector<Dir> dirs;
  for (std::size_t k = 0; k < d; ++k) {
    const int degree = d == 4 ? (k == 1 ? 2 : 1) : degrees[k];
    dirs.push_back(mixed_dir(degree, uniform[k]));
  }
  return dirs;
}

}  // namespace

TEST_CASE_TEMPLATE("SpanwiseElementExtraction: one-cell operations equal the dense reference", T,
                   double, float) {
  for (std::size_t d = 1; d <= 4; ++d) {
    const std::vector<Dir> dirs = dirs_for(d);
    for (const bs::ExtractionTarget target : kAllTargets) {
      const auto e = make_extraction<T>(dirs, target);
      const bs::BsplineSpace<T> space = bspline_test::space_of<T>(dirs);
      std::vector<pantr::NdArray<T>> full;
      for (const auto& s : space.spaces()) {
        full.push_back(operators_of<T>(s, target, LagrangeVariant::equispaces));
      }
      const std::vector<std::size_t> extents(e.num_intervals().begin(), e.num_intervals().end());
      // The corners, the middle, and a cell with an identity direction when there is one.
      const std::size_t total = e.num_total_intervals();
      for (const std::size_t flat : {std::size_t{0}, total / 2, total - 1, total / 3}) {
        const std::vector<std::size_t> cell = unravel(extents, flat);
        const std::vector<Dense> factors = factors_of_cell<T>(space, full, cell);
        for (const bs::ExtractionOp op : kAllOps) {
          CAPTURE(d);
          CAPTURE(static_cast<int>(target));
          CAPTURE(static_cast<int>(op));
          CAPTURE(flat);
          const std::size_t n = operand_entries<T>(e, op);
          const std::size_t side = is_vector_op(op) ? n : isqrt(n);
          const std::vector<T> operand = random_values<T>(n, 40 + static_cast<unsigned>(flat));
          const Dense operand_dense = dense_of<T>(operand, side, is_vector_op(op) ? 1 : side);
          const Reference ref = reference_of(op, factors, operand_dense);
          const std::vector<T> got = run_cell<T>(e, op, operand, cell);
          expect_within<T>(got, ref.result, ref.terms);
        }
      }
    }
  }
}

TEST_CASE_TEMPLATE("SpanwiseElementExtraction: kron_operator is the Kronecker product", T, double,
                   float) {
  for (std::size_t d = 1; d <= 4; ++d) {
    const std::vector<Dir> dirs = dirs_for(d);
    for (const bs::ExtractionTarget target : kAllTargets) {
      const auto e = make_extraction<T>(dirs, target);
      const bs::BsplineSpace<T> space = bspline_test::space_of<T>(dirs);
      std::vector<pantr::NdArray<T>> full;
      for (const auto& s : space.spaces()) {
        full.push_back(operators_of<T>(s, target, LagrangeVariant::equispaces));
      }
      const std::vector<std::size_t> extents(e.num_intervals().begin(), e.num_intervals().end());
      // Every cell, but every seventh in dimension 4 (400 cells of 24 x 24 exact entries).
      const std::size_t step = d == 4 ? 7 : 1;
      for (std::size_t flat = 0; flat < e.num_total_intervals(); flat += step) {
        const std::vector<std::size_t> cell = unravel(extents, flat);
        const Dense want = kron_all(factors_of_cell<T>(space, full, cell));
        pantr::NdArray<T> got({want.rows, want.cols}, pantr::for_overwrite);
        e.kron_operator(cell, got.template view<2>());
        // Each entry is a product of d factors multiplied left to right: d - 1 roundings.
        expect_within<T>(got.flat(), want, d - 1);
      }
    }
  }
}

TEST_CASE_TEMPLATE("SpanwiseElementExtraction: compact storage describes the 1D operators", T,
                   double, float) {
  for (std::size_t d = 1; d <= 3; ++d) {
    const std::vector<Dir> dirs = dirs_for(d);
    for (const bs::ExtractionTarget target : kAllTargets) {
      const auto e = make_extraction<T>(dirs, target);
      const bs::BsplineSpace<T> space = bspline_test::space_of<T>(dirs);
      CHECK(e.dim() == d);
      std::size_t identity_elements = 1;
      for (std::size_t k = 0; k < d; ++k) {
        const auto& s = space.spaces()[k];
        const auto order = static_cast<std::size_t>(s.degree()) + 1;
        const pantr::NdArray<T> ops = operators_of<T>(s, target, LagrangeVariant::equispaces);
        const auto n_el = static_cast<std::size_t>(s.num_intervals());
        CHECK(e.num_intervals()[k] == n_el);
        CHECK(e.input_shape_per_dir()[k] == order);
        CHECK(e.output_shape_per_dir()[k] == order);
        const auto mask = e.identity_mask(k);
        const auto idx = e.idx_map(k);
        const auto& compact = e.compact_ops(k);
        REQUIRE(mask.size() == n_el);
        REQUIRE(idx.size() == n_el);
        CHECK(compact.shape()[1] == order);
        CHECK(compact.shape()[2] == order);
        std::size_t n_identity = 0;
        std::size_t next_row = 0;
        for (std::size_t el = 0; el < n_el; ++el) {
          const T* full_op = ops.data() + el * order * order;
          if (mask[el]) {
            ++n_identity;
            // A flagged element's operator is the identity, exactly.
            for (std::size_t i = 0; i < order; ++i) {
              for (std::size_t j = 0; j < order; ++j) {
                CHECK(full_op[i * order + j] == (i == j ? T{1} : T{0}));
              }
            }
            CHECK(idx[el] == 0);
          } else {
            // The rows of the compact array are the non-identity operators, in order.
            REQUIRE(idx[el] == static_cast<std::int64_t>(next_row));
            REQUIRE(next_row < compact.shape()[0]);
            const T* row = compact.data() + next_row * order * order;
            CHECK(std::equal(row, row + order * order, full_op));
            ++next_row;
          }
          CHECK(idx[el] >= 0);
          CHECK(static_cast<std::size_t>(idx[el]) < compact.shape()[0]);
        }
        CHECK(compact.shape()[0] == (std::max)(n_el - n_identity, std::size_t{1}));
        identity_elements *= n_identity;
        // The cardinal mask is the space's own flags.
        if (target == bs::ExtractionTarget::cardinal) {
          std::unique_ptr<bool[]> cardinal(new bool[n_el]);
          s.cardinal_intervals(std::span<bool>(cardinal.get(), n_el));
          for (std::size_t el = 0; el < n_el; ++el) {
            CHECK(mask[el] == cardinal[el]);
          }
        }
      }
      CHECK(e.num_identity_elements() == identity_elements);
      CHECK(e.is_identity() == (identity_elements == e.num_total_intervals()));
    }
  }
}

TEST_CASE(
    "SpanwiseElementExtraction: the per-direction accessors refuse a direction out of range") {
  const auto e = make_extraction<double>(dirs_for(2), bs::ExtractionTarget::bezier);
  // An operation cast from an integer outside the four is refused too.
  const auto bad_op = static_cast<bs::ExtractionOp>(7);
  CHECK_THROWS_AS(static_cast<void>(e.required_scratch(bad_op)), pantr::value_error);
  CHECK_THROWS_AS(static_cast<void>(e.work_per_cell(bad_op)), pantr::value_error);
  CHECK_NOTHROW(static_cast<void>(e.compact_ops(1)));
  CHECK(message_of([&] { static_cast<void>(e.compact_ops(2)); }) ==
        "direction 2 out of range for 2 directions");
  CHECK_THROWS_AS(static_cast<void>(e.idx_map(2)), pantr::index_error);
  CHECK_THROWS_AS(static_cast<void>(e.identity_mask(99)), pantr::index_error);
}

TEST_CASE("SpanwiseElementExtraction: identity structure of the targets") {
  // A C^-1 space: every element of the Bezier target is a patch already.
  const Dir c_minus_1{{0, 0, 0, 1, 1, 1, 2, 2, 2}, 2, false};
  const auto bez = make_extraction<double>({c_minus_1, c_minus_1}, bs::ExtractionTarget::bezier);
  CHECK(bez.is_identity());
  CHECK(bez.num_identity_elements() == 4);
  // Degree 0: every element of every target is the 1x1 identity; the Lagrange operator of a
  // degree-0 space cannot be built, as before.
  const Dir deg0{{0, 1, 2, 3}, 0, false};
  CHECK(make_extraction<double>({deg0}, bs::ExtractionTarget::bezier).is_identity());
  CHECK(make_extraction<double>({deg0}, bs::ExtractionTarget::cardinal).is_identity());
  CHECK_THROWS_AS(make_extraction<double>({deg0}, bs::ExtractionTarget::lagrange),
                  pantr::value_error);
  // A smooth space: no Bezier element is an identity; the interior elements of a uniform one are
  // cardinal.
  const Dir smooth{bspline_test::clamped(2, {0.25, 0.5, 0.75}), 2, false};
  const auto smooth_bez = make_extraction<double>({smooth}, bs::ExtractionTarget::bezier);
  CHECK(smooth_bez.num_identity_elements() == 0);
  // Eight uniform elements of degree 2: the ones whose whole support is uniform are cardinal.
  const std::vector<double> eighths = {0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875};
  const Dir uniform8{bspline_test::clamped(2, eighths), 2, false};
  const auto uniform_card = make_extraction<double>({uniform8}, bs::ExtractionTarget::cardinal);
  CHECK(uniform_card.num_identity_elements() >= 1);
  CHECK(uniform_card.num_identity_elements() < 8);
  // Degree 1 Lagrange (equispaced) has the identity Lagrange-to-Bernstein matrix: the Bezier mask.
  const Dir lin{{0, 0, 1, 1, 2, 2, 3, 3}, 1, false};
  const auto lag = make_extraction<double>({lin}, bs::ExtractionTarget::lagrange);
  CHECK(lag.is_identity());
  // Degree 2 never.
  const auto lag2 = make_extraction<double>({c_minus_1}, bs::ExtractionTarget::lagrange);
  CHECK(lag2.num_identity_elements() == 0);
}

TEST_CASE("lagrange_structural_identity_mask: degree 0, degree 1 and higher") {
  const std::vector<double> k0 = {0, 1, 2, 3};
  const bs::BsplineSpace1D<double> deg0(std::span<const double>(k0), 0);
  std::unique_ptr<bool[]> flags(new bool[3]);
  bs::lagrange_structural_identity_mask<double>(deg0, LagrangeVariant::gauss_lobatto_legendre,
                                                std::span<bool>(flags.get(), 3));
  CHECK((flags[0] && flags[1] && flags[2]));
  // Degree 1, multiplicities [2, 2, 1, 2]: element 0 has multiplicity 2 at both ends, so it is a
  // patch; elements 1 and 2 have a simple knot at one end and are not.
  const std::vector<double> k1 = {0, 0, 1, 1, 2, 3, 3};
  const bs::BsplineSpace1D<double> deg1(std::span<const double>(k1), 1);
  for (const LagrangeVariant v :
       {LagrangeVariant::equispaces, LagrangeVariant::gauss_lobatto_legendre,
        LagrangeVariant::chebyshev_2nd}) {
    bs::lagrange_structural_identity_mask<double>(deg1, v, std::span<bool>(flags.get(), 3));
    CHECK((flags[0] && !flags[1] && !flags[2]));
  }
  // Gauss-Legendre nodes of degree 1 are not the Bernstein abscissae.
  bs::lagrange_structural_identity_mask<double>(deg1, LagrangeVariant::gauss_legendre,
                                                std::span<bool>(flags.get(), 3));
  CHECK((!flags[0] && !flags[1] && !flags[2]));
  CHECK(message_of([&] {
          bs::lagrange_structural_identity_mask<double>(deg1, LagrangeVariant::equispaces,
                                                        std::span<bool>(flags.get(), 2));
        }) == "Output array has shape (2,), but expected shape (3,)");
}

TEST_CASE("SpanwiseElementExtraction: a periodic direction is refused") {
  const Dir open{bspline_test::clamped(2, {0.5}), 2, false};
  const Dir periodic{bspline_test::periodic_knots(4, 2), 2, true};
  const char* expected =
      "SpanwiseElementExtraction does not yet support periodic directions. Convert the B-spline "
      "to open form first (see Bspline.to_open_bspline).";
  for (const bs::ExtractionTarget target : kAllTargets) {
    CHECK(message_of([&] { make_extraction<double>({periodic}, target); }) == expected);
    CHECK(message_of([&] { make_extraction<double>({open, periodic}, target); }) == expected);
    CHECK_THROWS_AS(make_extraction<float>({open, periodic}, target), pantr::not_implemented);
  }
  CHECK_THROWS_AS(static_cast<void>(bs::to_extraction_target(3)), pantr::value_error);
}

TEST_CASE("SpanwiseElementExtraction: the Lagrange variant selects the nodes") {
  const std::vector<Dir> dirs = {mixed_dir(3, 2)};
  const auto eq =
      make_extraction<double>(dirs, bs::ExtractionTarget::lagrange, LagrangeVariant::equispaces);
  const auto gll = make_extraction<double>(dirs, bs::ExtractionTarget::lagrange,
                                           LagrangeVariant::gauss_lobatto_legendre);
  CHECK(eq.lagrange_variant() == LagrangeVariant::equispaces);
  CHECK(gll.lagrange_variant() == LagrangeVariant::gauss_lobatto_legendre);
  const auto& a = eq.compact_ops(0);
  const auto& b = gll.compact_ops(0);
  REQUIRE(a.size() == b.size());
  double worst = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    worst = (std::max)(worst, std::abs(a.data()[i] - b.data()[i]));
  }
  // Degree 3: the two node sets differ, and so do the operators.
  CHECK(worst > 1e-3);
  CHECK(eq.target() == bs::ExtractionTarget::lagrange);
}

namespace {

/// A space of `d` directions with `n` uniform elements each, for the batch tests.
std::vector<Dir> uniform_dirs(std::size_t d, int degree, int n) {
  std::vector<Dir> dirs;
  for (std::size_t k = 0; k < d; ++k) {
    std::vector<double> interior;
    for (int i = 1; i < n; ++i) {
      interior.push_back(static_cast<double>(i) / static_cast<double>(n));
    }
    dirs.push_back(Dir{bspline_test::clamped(degree, interior), degree, false});
  }
  return dirs;
}

/// The cells `0 .. total-1` repeated `cycles` times, as `(n, d)` per-direction indices.
std::vector<std::int64_t> all_cells(const std::vector<std::size_t>& extents, std::size_t cycles) {
  std::size_t total = 1;
  for (const std::size_t n : extents) {
    total *= n;
  }
  std::vector<std::int64_t> cells;
  for (std::size_t c = 0; c < cycles; ++c) {
    for (std::size_t flat = 0; flat < total; ++flat) {
      for (const std::size_t i : unravel(extents, flat)) {
        cells.push_back(static_cast<std::int64_t>(i));
      }
    }
  }
  return cells;
}

/// Run a batch of one operation; returns the flat results.
template <class T>
std::vector<T> run_batch(const bs::SpanwiseElementExtraction<T>& e, bs::ExtractionOp op,
                         const std::vector<T>& operand, const std::vector<std::int64_t>& cells,
                         std::optional<pantr::mdview<T, 2>> scratch = std::nullopt) {
  const std::size_t d = e.dim();
  const std::size_t n_cells = cells.size() / d;
  std::size_t n_in = 1;
  std::size_t n_out = 1;
  for (std::size_t k = 0; k < d; ++k) {
    n_in *= e.input_shape_per_dir()[k];
    n_out *= e.output_shape_per_dir()[k];
  }
  const pantr::mdview<const std::int64_t, 2> cv(cells.data(), n_cells, d);
  switch (op) {
    case bs::ExtractionOp::apply: {
      std::vector<T> out(n_cells * n_out);
      e.apply_many(pantr::mdview<const T, 2>(operand.data(), n_cells, n_in), cv,
                   pantr::mdview<T, 2>(out.data(), n_cells, n_out), scratch);
      return out;
    }
    case bs::ExtractionOp::apply_transpose: {
      std::vector<T> out(n_cells * n_in);
      e.apply_transpose_many(pantr::mdview<const T, 2>(operand.data(), n_cells, n_out), cv,
                             pantr::mdview<T, 2>(out.data(), n_cells, n_in), scratch);
      return out;
    }
    case bs::ExtractionOp::mt_k_m: {
      std::vector<T> out(n_cells * n_in * n_in);
      e.apply_mt_k_m_many(pantr::mdview<const T, 3>(operand.data(), n_cells, n_out, n_out), cv,
                          pantr::mdview<T, 3>(out.data(), n_cells, n_in, n_in), scratch);
      return out;
    }
    case bs::ExtractionOp::m_k_mt: {
      std::vector<T> out(n_cells * n_out * n_out);
      e.apply_m_k_mt_many(pantr::mdview<const T, 3>(operand.data(), n_cells, n_in, n_in), cv,
                          pantr::mdview<T, 3>(out.data(), n_cells, n_out, n_out), scratch);
      return out;
    }
  }
  return {};
}

/// The magnitudes of the terms of an operation on one cell: the reference of the operation on
/// the entrywise absolute values of the operator and of the operand (`want.a` of its result), and
/// the number of terms of its stages. Operands and operators are bounded by 1 in magnitude, so
/// the operand's magnitudes are taken as 1.
template <class T>
Reference magnitudes_of_cell(const bs::SpanwiseElementExtraction<T>& e, bs::ExtractionOp op,
                             const std::vector<std::size_t>& cell) {
  const std::size_t d = e.dim();
  std::vector<Dense> factors;
  for (std::size_t k = 0; k < d; ++k) {
    // The factor of the cell, from the 1D operator of the interval: its magnitudes.
    const auto order = e.input_shape_per_dir()[k];
    const auto& dir_ops = e.compact_ops(k);
    const bool ident = e.identity_mask(k)[cell[k]];
    if (ident) {
      factors.push_back(identity_dense(order, false));
    } else {
      factors.push_back(dense_of<T>(
          std::span<const T>(
              dir_ops.data() + static_cast<std::size_t>(e.idx_map(k)[cell[k]]) * order * order,
              order * order),
          order, order, false));
    }
  }
  const std::size_t n = operand_entries<T>(e, op);
  const std::size_t side = is_vector_op(op) ? n : isqrt(n);
  Dense ones = make_dense(side, is_vector_op(op) ? 1 : side, false);
  std::fill(ones.a.begin(), ones.a.end(), 1.0);
  return reference_of(op, factors, ones);
}

}  // namespace

TEST_CASE_TEMPLATE("SpanwiseElementExtraction: a batch equals the one-cell operations", T, double,
                   float) {
  for (std::size_t d = 1; d <= 3; ++d) {
    const std::vector<Dir> dirs =
        d == 3 ? std::vector<Dir>{mixed_dir(2, 2), mixed_dir(3, 1), mixed_dir(1, 2)} : dirs_for(d);
    for (const bs::ExtractionTarget target :
         {bs::ExtractionTarget::bezier, bs::ExtractionTarget::lagrange,
          bs::ExtractionTarget::cardinal}) {
      const auto e = make_extraction<T>(dirs, target);
      const std::vector<std::size_t> extents(e.num_intervals().begin(), e.num_intervals().end());
      const std::vector<std::int64_t> cells = all_cells(extents, 2);
      const std::size_t n_cells = cells.size() / d;
      for (const bs::ExtractionOp op : kAllOps) {
        CAPTURE(d);
        CAPTURE(static_cast<int>(target));
        CAPTURE(static_cast<int>(op));
        const std::size_t per_cell = operand_entries<T>(e, op);
        const std::vector<T> operand = random_values<T>(n_cells * per_cell, 77);
        const std::vector<T> batch = run_batch<T>(e, op, operand, cells);
        // With a caller scratch row per cell, exactly as wide as needed.
        const std::size_t width = (std::max)(e.required_scratch(op), std::size_t{1});
        std::vector<T> scratch_data(n_cells * width);
        const std::vector<T> batch_scratch = run_batch<T>(
            e, op, operand, cells, pantr::mdview<T, 2>(scratch_data.data(), n_cells, width));
        REQUIRE(batch.size() == batch_scratch.size());
        const std::size_t out_per_cell = batch.size() / n_cells;
        for (std::size_t c = 0; c < n_cells; ++c) {
          std::vector<std::size_t> cell(d);
          for (std::size_t k = 0; k < d; ++k) {
            cell[k] = static_cast<std::size_t>(cells[c * d + k]);
          }
          const auto first = operand.begin() + static_cast<std::ptrdiff_t>(c * per_cell);
          const std::vector<T> one = run_cell<T>(
              e, op, std::vector<T>(first, first + static_cast<std::ptrdiff_t>(per_cell)), cell);
          REQUIRE(one.size() == out_per_cell);
          const Reference mag = magnitudes_of_cell<T>(e, op, cell);
          const auto at = static_cast<std::ptrdiff_t>(c * out_per_cell);
          const std::span<const T> in_batch(batch.data() + at, out_per_cell);
          const std::span<const T> in_scratch(batch_scratch.data() + at, out_per_cell);
          expect_agree<T>(in_batch, one, mag.result, mag.terms);
          expect_agree<T>(in_scratch, one, mag.result, mag.terms);
        }
      }
    }
  }
}

TEST_CASE("SpanwiseElementExtraction: a batch is the same under any number of threads") {
  using T = double;
  const std::vector<Dir> dirs = uniform_dirs(3, 3, 5);
  const auto e = make_extraction<T>(dirs, bs::ExtractionTarget::bezier);
  const std::vector<std::size_t> extents(e.num_intervals().begin(), e.num_intervals().end());
  const std::size_t total = e.num_total_intervals();
  // Enough work to cross the threshold and spread over several chunks.
  const std::vector<std::int64_t> cells = all_cells(extents, 12);
  const std::size_t n_cells = cells.size() / 3;
  const std::int64_t saved = static_cast<std::int64_t>(pantr::parallel::get_num_threads());
  for (const bs::ExtractionOp op : kAllOps) {
    const std::size_t per_cell = operand_entries<T>(e, op);
    const std::vector<T> operand = random_values<T>(n_cells * per_cell, 5);
    pantr::parallel::set_num_threads(1);
    const std::vector<T> serial = run_batch<T>(e, op, operand, cells);
    pantr::parallel::set_num_threads(static_cast<std::int64_t>(pantr::parallel::max_threads()));
    const std::vector<T> parallel = run_batch<T>(e, op, operand, cells);
    // The caller's scratch, a row per cell, under the same threads: no row is shared.
    const std::size_t width = (std::max)(e.required_scratch(op), std::size_t{1});
    std::vector<T> rows(n_cells * width);
    const std::vector<T> parallel_rows =
        run_batch<T>(e, op, operand, cells, pantr::mdview<T, 2>(rows.data(), n_cells, width));
    pantr::parallel::set_num_threads(saved);
    REQUIRE(serial.size() == parallel.size());
    REQUIRE(serial.size() == parallel_rows.size());
    const std::size_t out_per_cell = serial.size() / n_cells;
    // A cell is computed by one thread in the same order of operations whatever the chunking: the
    // two runs agree within twice the derived bound. Every seventh cell is checked; 7 is coprime
    // with the 125 elements, so every element is visited.
    for (std::size_t c = 0; c < n_cells; c += 7) {
      const std::vector<std::size_t> cell = unravel(extents, c % total);
      const Reference mag = magnitudes_of_cell<T>(e, op, cell);
      const auto at = static_cast<std::ptrdiff_t>(c * out_per_cell);
      expect_agree<T>(std::span<const T>(serial.data() + at, out_per_cell),
                      std::span<const T>(parallel.data() + at, out_per_cell), mag.result,
                      mag.terms);
      expect_agree<T>(std::span<const T>(serial.data() + at, out_per_cell),
                      std::span<const T>(parallel_rows.data() + at, out_per_cell), mag.result,
                      mag.terms);
    }
  }
}

TEST_CASE("SpanwiseElementExtraction: concurrent applications of one value agree") {
  const std::vector<Dir> dirs = uniform_dirs(2, 3, 6);
  const auto e = make_extraction<double>(dirs, bs::ExtractionTarget::bezier);
  const std::vector<std::size_t> extents(e.num_intervals().begin(), e.num_intervals().end());
  const std::vector<std::int64_t> cells = all_cells(extents, 4);
  const std::size_t n_cells = cells.size() / 2;
  const std::size_t per_cell = operand_entries<double>(e, bs::ExtractionOp::mt_k_m);
  const std::vector<double> operand = random_values<double>(n_cells * per_cell, 3);
  // The same deterministic computation, run by several threads at once on one shared value (and
  // inside it, the pool): each run is the serial one, because a call reads the shared operators and
  // writes only its own buffers. This is what TSan watches; the values are compared exactly
  // because it is one function on the same data in one binary, not a cross-toolchain claim.
  const std::vector<double> expected =
      run_batch<double>(e, bs::ExtractionOp::mt_k_m, operand, cells);
  const std::vector<std::size_t> one_cell = {2, 3};
  const std::vector<double> single_operand(operand.begin(),
                                           operand.begin() + static_cast<std::ptrdiff_t>(per_cell));
  const std::vector<double> single_expected =
      run_cell<double>(e, bs::ExtractionOp::mt_k_m, single_operand, one_cell);
  const std::size_t n_threads = 6;
  std::vector<std::vector<double>> results(n_threads);
  std::vector<std::vector<double>> singles(n_threads);
  std::vector<std::thread> threads;
  for (std::size_t t = 0; t < n_threads; ++t) {
    threads.emplace_back([&, t] {
      for (int rep = 0; rep < 3; ++rep) {
        results[t] = run_batch<double>(e, bs::ExtractionOp::mt_k_m, operand, cells);
        singles[t] = run_cell<double>(e, bs::ExtractionOp::mt_k_m, single_operand, one_cell);
      }
    });
  }
  for (std::thread& th : threads) {
    th.join();
  }
  for (std::size_t t = 0; t < n_threads; ++t) {
    CHECK(results[t] == expected);
    CHECK(singles[t] == single_expected);
  }
}

TEST_CASE("SpanwiseElementExtraction: scratch of exactly the required size is enough") {
  // Heap buffers of exactly the size asked for: an overrun is a heap overflow the sanitizer sees.
  for (std::size_t d = 1; d <= 4; ++d) {
    const std::vector<Dir> dirs = dirs_for(d);
    const auto e = make_extraction<double>(dirs, bs::ExtractionTarget::bezier);
    // The second element of each direction is smooth: no direction is skipped.
    const std::vector<std::size_t> cell(d, 1);
    for (const bs::ExtractionOp op : kAllOps) {
      const std::size_t need = e.required_scratch(op);
      const std::size_t n = operand_entries<double>(e, op);
      const std::vector<double> operand = random_values<double>(n, 13);
      const std::vector<double> baseline = run_cell<double>(e, op, operand, cell);
      const auto run_with = [&](std::span<double> scratch) {
        std::vector<double> out(baseline.size());
        const bool matrix = !is_vector_op(op);
        const std::size_t side = matrix ? isqrt(n) : 0;
        const std::size_t oside = matrix ? isqrt(out.size()) : 0;
        switch (op) {
          case bs::ExtractionOp::apply:
            e.apply(operand, cell, out, scratch);
            break;
          case bs::ExtractionOp::apply_transpose:
            e.apply_transpose(operand, cell, out, scratch);
            break;
          case bs::ExtractionOp::mt_k_m:
            e.apply_mt_k_m(pantr::mdview<const double, 2>(operand.data(), side, side), cell,
                           pantr::mdview<double, 2>(out.data(), oside, oside), scratch);
            break;
          case bs::ExtractionOp::m_k_mt:
            e.apply_m_k_mt(pantr::mdview<const double, 2>(operand.data(), side, side), cell,
                           pantr::mdview<double, 2>(out.data(), oside, oside), scratch);
            break;
        }
        return out;
      };
      CAPTURE(d);
      CAPTURE(static_cast<int>(op));
      const std::unique_ptr<double[]> exact_scratch(new double[need == 0 ? 1 : need]);
      CHECK(run_with(std::span<double>(exact_scratch.get(), need)) == baseline);
      if (need > 0) {
        CHECK(message_of([&] {
                (void)run_with(std::span<double>(exact_scratch.get(), need - 1));
              }) == "Scratch size " + std::to_string(need - 1) + " is smaller than required " +
                        std::to_string(need));
      }
    }
  }
}

TEST_CASE(
    "SpanwiseElementExtraction: refuses bad arguments, with the messages of the Python class") {
  const std::vector<Dir> dirs = {mixed_dir(2, 2), mixed_dir(2, 2)};
  const auto e = make_extraction<double>(dirs, bs::ExtractionTarget::bezier);
  const std::size_t n_in = 9;
  std::vector<double> v(n_in, 1.0);
  std::vector<double> out(n_in);
  const std::vector<std::size_t> cell = {1, 1};
  // The cell.
  CHECK(message_of([&] { e.apply(v, std::vector<std::size_t>{1}, out); }) ==
        "Per-direction cell index has length 1, expected 2");
  CHECK(message_of([&] { e.apply(v, std::vector<std::size_t>{1, 99}, out); }) ==
        "Cell index 99 out of range for direction 1 with 4 elements");
  CHECK_THROWS_AS(e.apply(v, std::vector<std::size_t>{1, 99}, out), pantr::index_error);
  CHECK(message_of([&] { (void)e.unravel(16); }) ==
        "Flat cell index 16 out of range for 16 elements");
  CHECK(e.unravel(7) == std::vector<std::size_t>({1, 3}));
  // The operand and the result.
  CHECK(message_of([&] { e.apply(std::span<const double>(v).first(8), cell, out); }) ==
        "Operand has shape (8,), but expected shape (9,)");
  CHECK(message_of([&] { e.apply(v, cell, std::span<double>(out).first(8)); }) ==
        "Output array has shape (8,), but expected shape (9,)");
  std::vector<double> k(81);
  std::vector<double> o(81);
  CHECK(message_of([&] {
          e.apply_mt_k_m(pantr::mdview<const double, 2>(k.data(), 9, 8), cell,
                         pantr::mdview<double, 2>(o.data(), 9, 9));
        }) == "Operand has shape (9, 8), but expected shape (9, 9)");
  CHECK(message_of([&] {
          e.apply_m_k_mt(pantr::mdview<const double, 2>(k.data(), 9, 9), cell,
                         pantr::mdview<double, 2>(o.data(), 3, 27));
        }) == "Output array has shape (3, 27), but expected shape (9, 9)");
  // The scratch, and the overlaps.
  const std::size_t need = e.required_scratch(bs::ExtractionOp::apply);
  REQUIRE(need > 0);
  std::vector<double> too_short(need - 1);
  CHECK(message_of([&] { e.apply(v, cell, out, std::span<double>(too_short)); }) ==
        "Scratch size " + std::to_string(need - 1) + " is smaller than required " +
            std::to_string(need));
  std::vector<double> buf(100, 1.0);
  CHECK(message_of([&] {
          e.apply(std::span<const double>(buf.data(), 9), cell,
                  std::span<double>(buf.data() + 3, 9));
        }) == "out must not alias the operand; the kernel reads and writes overlapping memory");
  CHECK(message_of([&] {
          e.apply(std::span<const double>(buf.data(), 9), cell,
                  std::span<double>(buf.data() + 9, 9), std::span<double>(buf.data() + 10, need));
        }) == "scratch must not overlap the operand or out");
  // A cell that is the identity in every direction may overwrite its operand (the element (0, 0) of
  // a C^-1 space).
  const Dir c_minus_1{{0, 0, 0, 1, 1, 1, 2, 2, 2}, 2, false};
  const auto ident = make_extraction<double>({c_minus_1, c_minus_1}, bs::ExtractionTarget::bezier);
  std::vector<double> same(9, 2.0);
  ident.apply(same, std::vector<std::size_t>{0, 1}, same);
  CHECK(same == std::vector<double>(9, 2.0));
  // kron_operator.
  pantr::NdArray<double> op({8, 9});
  CHECK(message_of([&] { e.kron_operator(cell, op.template view<2>()); }) ==
        "Output array has shape (8, 9), but expected shape (9, 9)");
}

TEST_CASE("SpanwiseElementExtraction: a batch refuses bad arguments") {
  const std::vector<Dir> dirs = {mixed_dir(2, 2), mixed_dir(2, 2)};
  const auto e = make_extraction<double>(dirs, bs::ExtractionTarget::bezier);
  const std::vector<std::int64_t> cells = {0, 0, 1, 1, 3, 2};
  const std::size_t n = 3;
  std::vector<double> v(n * 9, 1.0);
  std::vector<double> out(n * 9);
  const auto v2 = [&](std::size_t rows, std::size_t cols) {
    return pantr::mdview<const double, 2>(v.data(), rows, cols);
  };
  const auto o2 = [&](std::size_t rows, std::size_t cols) {
    return pantr::mdview<double, 2>(out.data(), rows, cols);
  };
  const pantr::mdview<const std::int64_t, 2> cv(cells.data(), n, 2);
  CHECK_NOTHROW(e.apply_many(v2(n, 9), cv, o2(n, 9)));
  CHECK(message_of([&] { e.apply_many(v2(n, 9), cv, o2(n, 8)); }) ==
        "Output array has shape (3, 8), but expected shape (3, 9)");
  CHECK(message_of([&] { e.apply_many(v2(2, 9), cv, o2(n, 9)); }) ==
        "Operand has shape (2, 9), but expected shape (3, 9)");
  CHECK(message_of([&] {
          e.apply_many(v2(n, 9), pantr::mdview<const std::int64_t, 2>(cells.data(), 2, 3),
                       o2(n, 9));
        }) == "cell_indices.shape[1]=3 does not match d=2");
  const std::vector<std::int64_t> bad = {0, 0, 1, 1, 4, 2};
  CHECK(message_of([&] {
          e.apply_many(v2(n, 9), pantr::mdview<const std::int64_t, 2>(bad.data(), n, 2), o2(n, 9));
        }) == "cell_indices[:, 0] contains values outside [0, 4)");
  const std::vector<std::int64_t> negative = {0, 0, 1, -1, 3, 2};
  CHECK_THROWS_AS(
      e.apply_many(v2(n, 9), pantr::mdview<const std::int64_t, 2>(negative.data(), n, 2), o2(n, 9)),
      pantr::index_error);
  const std::size_t width = (std::max)(e.required_scratch(bs::ExtractionOp::apply), std::size_t{1});
  std::vector<double> sdata(n * width);
  CHECK(message_of([&] {
          e.apply_many(v2(n, 9), cv, o2(n, 9),
                       pantr::mdview<double, 2>(sdata.data(), n + 1, width));
        }) == "Batch scratch shape[0]=4 does not match n_cells=3");
  CHECK(message_of([&] {
          e.apply_many(v2(n, 9), cv, o2(n, 9),
                       pantr::mdview<double, 2>(sdata.data(), n, width - 1));
        }) == "Batch scratch shape[1]=" + std::to_string(width - 1) +
                  " is smaller than required width=" + std::to_string(width));
  CHECK(message_of([&] { e.apply_many(v2(n, 9), cv, pantr::mdview<double, 2>(v.data(), n, 9)); }) ==
        "out must not alias the operand; the kernel reads and writes overlapping memory");
  std::vector<double> overlap(n * width + n * 9 + n * 9, 1.0);
  CHECK(message_of([&] {
          e.apply_many(pantr::mdview<const double, 2>(overlap.data(), n, 9), cv,
                       pantr::mdview<double, 2>(overlap.data() + n * 9, n, 9),
                       pantr::mdview<double, 2>(overlap.data() + n * 9 + 2, n, width));
        }) == "scratch must not overlap the operand or out");
  // Every cell identity: the batch may overwrite its operand.
  const Dir c_minus_1{{0, 0, 0, 1, 1, 1, 2, 2, 2}, 2, false};
  const auto ident = make_extraction<double>({c_minus_1, c_minus_1}, bs::ExtractionTarget::bezier);
  const std::vector<std::int64_t> any = {0, 0, 1, 1};
  std::vector<double> same(18, 3.0);
  ident.apply_many(pantr::mdview<const double, 2>(same.data(), 2, 9),
                   pantr::mdview<const std::int64_t, 2>(any.data(), 2, 2),
                   pantr::mdview<double, 2>(same.data(), 2, 9));
  CHECK(same == std::vector<double>(18, 3.0));
  // No cells: nothing to do, and the shapes are still checked.
  std::vector<double> none_v;
  std::vector<double> none_o;
  const pantr::mdview<const std::int64_t, 2> no_cells(cells.data(), 0, 2);
  CHECK_NOTHROW(e.apply_many(pantr::mdview<const double, 2>(none_v.data(), 0, 9), no_cells,
                             pantr::mdview<double, 2>(none_o.data(), 0, 9)));
  CHECK_NOTHROW(e.apply_m_k_mt_many(pantr::mdview<const double, 3>(none_v.data(), 0, 9, 9),
                                    no_cells, pantr::mdview<double, 3>(none_o.data(), 0, 9, 9)));
  CHECK(message_of([&] {
          e.apply_many(pantr::mdview<const double, 2>(none_v.data(), 0, 9), no_cells,
                       pantr::mdview<double, 2>(none_o.data(), 0, 8));
        }) == "Output array has shape (0, 8), but expected shape (0, 9)");
  CHECK(message_of([&] {
          e.apply_many(pantr::mdview<const double, 2>(none_v.data(), 0, 9), no_cells,
                       pantr::mdview<double, 2>(none_o.data(), 0, 9),
                       pantr::mdview<double, 2>(sdata.data(), 0, 1));
        }) == "Batch scratch shape[1]=1 is smaller than required width=" + std::to_string(width));
}
