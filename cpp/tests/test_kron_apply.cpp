// The Kronecker application behind SpanwiseElementExtraction (`apply_kron`) on arbitrary factors,
// and the scratch size it needs.
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
#include <doctest/doctest.h>

#include "bspline_oracle.hpp"
#include "bspline_test_support.hpp"
#include "spanwise_extraction_support.hpp"

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
using namespace spanwise_test;

namespace {

/// One direction's factor shape and flag.
struct Shape {
  std::size_t rows;
  std::size_t cols;
  bool identity;
};

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
