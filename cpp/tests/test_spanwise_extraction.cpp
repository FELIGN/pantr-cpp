// SpanwiseElementExtraction on one cell: construction, the compact storage and the identity
// structure of each target, the Lagrange variant, and the one-cell operations against the dense
// Kronecker reference (the oracle and the bound are those of test_kron_apply.cpp).
// The per-direction operators of the class are compared with the 1D operator functions of
// `extraction.hpp` (tested in test_bspline_extraction.cpp), not with a recomputation of the class.
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
