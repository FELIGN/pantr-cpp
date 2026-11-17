// What SpanwiseElementExtraction and `apply_kron` refuse, and with which message (the Python
// class's messages, which the Python suite matches): bad cells, shapes, scratch, overlapping
// buffers, periodic directions, directions out of range.
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
  // Every cell identity: the batch may overwrite its operand, as the very same range (a shifted
  // overlap is refused, below).
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

TEST_CASE(
    "SpanwiseElementExtraction: a batch of identity cells overwrites its operand only in place") {
  // The cells of a batch run on several threads. Copies over their own operand are safe (cell `c`
  // reads and writes only its own entries); an `out` shifted against the operand, by a cell or by
  // one entry, makes a chunk write what another one reads, so the result would depend on the
  // chunking. The batch is long enough to be cut into several chunks.
  const Dir c_minus_1{{0, 0, 0, 1, 1, 1, 2, 2, 2}, 2, false};
  const auto ident = make_extraction<double>({c_minus_1, c_minus_1}, bs::ExtractionTarget::bezier);
  const std::size_t n = 4096;
  std::vector<std::int64_t> cells(n * 2);
  for (std::size_t c = 0; c < n; ++c) {
    cells[c * 2] = static_cast<std::int64_t>(c % 2);
    cells[c * 2 + 1] = static_cast<std::int64_t>((c / 2) % 2);
  }
  const pantr::mdview<const std::int64_t, 2> cv(cells.data(), n, 2);
  const char* const refusal =
      "out must not alias the operand; the kernel reads and writes overlapping memory";

  std::vector<double> buf = random_values<double>(n * 9 + 9, 31);
  const std::vector<double> kept = buf;
  const auto in = [&] { return pantr::mdview<const double, 2>(buf.data(), n, 9); };
  const auto out_at = [&](std::size_t offset) {
    return pantr::mdview<double, 2>(buf.data() + offset, n, 9);
  };
  // The very same range: a copy onto itself.
  CHECK_NOTHROW(ident.apply_many(in(), cv, out_at(0)));
  CHECK(buf == kept);
  // Shifted by a cell, and by one entry: refused, and nothing is written.
  CHECK(message_of([&] { ident.apply_many(in(), cv, out_at(9)); }) == refusal);
  CHECK(buf == kept);
  CHECK(message_of([&] { ident.apply_many(in(), cv, out_at(1)); }) == refusal);
  CHECK(buf == kept);
  // The matrix operations, whose cells are 81 entries.
  std::vector<double> mats = random_values<double>((n + 1) * 81, 33);
  const std::vector<double> mats_kept = mats;
  CHECK_NOTHROW(ident.apply_mt_k_m_many(pantr::mdview<const double, 3>(mats.data(), n, 9, 9), cv,
                                        pantr::mdview<double, 3>(mats.data(), n, 9, 9)));
  CHECK(mats == mats_kept);
  CHECK(message_of([&] {
          ident.apply_mt_k_m_many(pantr::mdview<const double, 3>(mats.data(), n, 9, 9), cv,
                                  pantr::mdview<double, 3>(mats.data() + 81, n, 9, 9));
        }) == refusal);
  CHECK(message_of([&] {
          ident.apply_m_k_mt_many(pantr::mdview<const double, 3>(mats.data(), n, 9, 9), cv,
                                  pantr::mdview<double, 3>(mats.data() + 1, n, 9, 9));
        }) == refusal);
  CHECK(mats == mats_kept);
}

TEST_CASE("SpanwiseElementExtraction: a batch with a non-identity cell refuses a shifted overlap") {
  const std::vector<Dir> dirs = {mixed_dir(2, 2), mixed_dir(2, 2)};
  const auto e = make_extraction<double>(dirs, bs::ExtractionTarget::bezier);
  const std::size_t n = 64;
  std::vector<std::int64_t> cells(n * 2, 1);  // the element (1, 1) is smooth
  std::vector<double> buf = random_values<double>(n * 9 + 9, 35);
  const std::vector<double> kept = buf;
  const char* const refusal =
      "out must not alias the operand; the kernel reads and writes overlapping memory";
  CHECK(message_of([&] {
          e.apply_many(pantr::mdview<const double, 2>(buf.data(), n, 9),
                       pantr::mdview<const std::int64_t, 2>(cells.data(), n, 2),
                       pantr::mdview<double, 2>(buf.data() + 9, n, 9));
        }) == refusal);
  CHECK(buf == kept);
}

TEST_CASE(
    "SpanwiseElementExtraction: one identity cell is a copy for any overlap with its operand") {
  // One cell is one `memmove` on one thread, so a shifted overlap has a defined result (unlike a
  // batch, whose cells run on several threads).
  const Dir c_minus_1{{0, 0, 0, 1, 1, 1, 2, 2, 2}, 2, false};
  const auto ident = make_extraction<double>({c_minus_1, c_minus_1}, bs::ExtractionTarget::bezier);
  std::vector<double> buf = random_values<double>(10, 37);
  const std::vector<double> operand(buf.begin(), buf.begin() + 9);
  ident.apply(std::span<const double>(buf.data(), 9), std::vector<std::size_t>{0, 1},
              std::span<double>(buf.data() + 1, 9));
  CHECK(std::vector<double>(buf.begin() + 1, buf.end()) == operand);
  // A cell that is not the identity in every direction still refuses it.
  const auto e =
      make_extraction<double>({mixed_dir(2, 2), mixed_dir(2, 2)}, bs::ExtractionTarget::bezier);
  CHECK(message_of([&] {
          e.apply(std::span<const double>(buf.data(), 9), std::vector<std::size_t>{1, 1},
                  std::span<double>(buf.data() + 1, 9));
        }) == "out must not alias the operand; the kernel reads and writes overlapping memory");
}

TEST_CASE(
    "SpanwiseElementExtraction: a space of many directions is refused for the size of its "
    "operand") {
  // Two-sided operations hold `prod(degree_k + 1)^2` entries: 33 directions of degree 1 are 2^66.
  // The space accepts them (`prod(degree_k + 1) = 2^33`); the message names the number of
  // directions, not the 66 extents of the product.
  const Dir line{{0, 0, 1, 1}, 1, false};
  const std::vector<Dir> dirs(33, line);
  const std::string message =
      message_of([&] { make_extraction<double>(dirs, bs::ExtractionTarget::bezier); });
  CHECK(message ==
        "the size of the operand or of an intermediate result for d = 33 directions is too large "
        "to count");
  CHECK_THROWS_AS(make_extraction<float>(dirs, bs::ExtractionTarget::bezier),
                  pantr::capacity_error);
}
