// SpanwiseElementExtraction on a batch of cells: a batch equals the one-cell operations within the
// derived bound of test_kron_apply.cpp, whatever the number of threads, and concurrent
// applications of one value agree.
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
