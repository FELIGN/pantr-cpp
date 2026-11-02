// MultiLevelExtraction: the per-cell operators `M` (level-L B-splines to the active functions) and
// `M E` (target basis to the active functions). Oracles, each from a route other than the code
// under test:
//   - `sum_c (M E)(j, c) B_c(xi)` against the space's own `tabulate_basis` at the mapped point, the
//     reference basis `B` tabulated by its closed form (the Bernstein products
//     `binom(p, i) xi^i (1 - xi)^(p - i)` and the equispaced Lagrange products), never through the
//     library's basis kernels;
//   - the columns of `M E` of a truncated space sum to one (the basis is a partition of unity and
//     the reference bases are linearly independent and sum to one), with the plain hierarchical
//     basis as the control that does not;
//   - an unrefined space, whose `M` is the identity and whose `M E` is the single-level operator;
//   - values recorded from the Python implementation (thb_extraction_golden.hpp says how).
#include <doctest/doctest.h>

#include "thb_extraction_golden.hpp"
#include "thb_test_support.hpp"

#include <pantr/bspline/spanwise_extraction.hpp>
#include <pantr/core/errors.hpp>
#include <pantr/core/mdspan.hpp>
#include <pantr/core/ndarray.hpp>
#include <pantr/thb/multilevel_extraction.hpp>
#include <pantr/thb/space.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace {

namespace bs = pantr::bspline;
namespace gold = pantr_thb_extraction_golden;
using pantr::mdview;
using pantr::NdArray;
using pantr::thb::MultiLevelExtraction;
using pantr::thb::THBSplineSpace;
using thb_test::build_slack;
using thb_test::eps;
using thb_test::I64;
using thb_test::Ivec;
using thb_test::k3;
using thb_test::value_error_units;
using Space = THBSplineSpace<double>;
using Target = bs::ExtractionTarget;

// ---- Spaces ----------------------------------------------------------------------------------

/// The spaces of the comparisons: the 2D three-level space (truncated and plain), a 3D one, a 1D
/// cubic with regularity 0, a 2D space of mixed degrees (2, 3), and one whose factor is 1 along the
/// first axis (that direction is never subdivided: an identity two-scale matrix).
std::vector<Space> test_spaces() {
  std::vector<Space> out;
  out.push_back(thb_test::refined_2d(true));
  out.push_back(thb_test::refined_2d(false));
  const auto b3 =
      thb_test::make_thb_space({thb_test::k2_3d, thb_test::k2_3d, thb_test::k2_3d}, {2, 2, 2},
                               {2, 2, 2}, {std::nullopt, std::nullopt, std::nullopt}, true);
  out.push_back(b3.refine(Ivec{0, 3}, std::nullopt));
  const auto b1 = thb_test::make_thb_space({k3}, {3}, {2}, {0}, true);
  out.push_back(b1.refine_region(0, Ivec{0}, Ivec{2}, std::nullopt)
                    .refine_region(1, Ivec{0}, Ivec{2}, std::nullopt));
  const auto mixed = thb_test::make_thb_space({thb_test::k2, k3}, {2, 3}, {2, 2},
                                              {std::nullopt, std::nullopt}, true);
  out.push_back(mixed.refine_region(0, Ivec{0, 0}, Ivec{2, 2}, std::nullopt));
  const auto aniso = thb_test::make_thb_space({thb_test::k2, thb_test::k2}, {2, 2}, {1, 2},
                                              {std::nullopt, std::nullopt}, true);
  out.push_back(aniso.refine(Ivec{0}, std::nullopt));
  return out;
}

// ---- The reference bases in closed form ------------------------------------------------------

/// The 1D basis of `target` of degree `p` at `xi`: Bernstein, or Lagrange at the equispaced nodes
/// `j / p`.
std::vector<double> basis_1d(Target target, int p, double xi) {
  std::vector<double> b(static_cast<std::size_t>(p) + 1, 1.0);
  if (target == Target::bezier) {
    double binom = 1.0;
    for (int i = 0; i <= p; ++i) {
      double v = binom;
      for (int a = 0; a < i; ++a) {
        v *= xi;
      }
      for (int a = 0; a < p - i; ++a) {
        v *= 1.0 - xi;
      }
      b[static_cast<std::size_t>(i)] = v;
      binom = binom * static_cast<double>(p - i) / static_cast<double>(i + 1);
    }
  } else {
    for (int i = 0; i <= p; ++i) {
      for (int j = 0; j <= p; ++j) {
        if (j != i) {
          b[static_cast<std::size_t>(i)] *= (xi - static_cast<double>(j) / p) /
                                            (static_cast<double>(i - j) / static_cast<double>(p));
        }
      }
    }
  }
  return b;
}

/// The tensor-product basis at `xi`, C order (the last direction fastest).
std::vector<double> basis_nd(Target target, const std::vector<int>& degrees, const double* xi) {
  std::vector<double> acc = {1.0};
  for (std::size_t k = 0; k < degrees.size(); ++k) {
    const std::vector<double> one = basis_1d(target, degrees[k], xi[k]);
    std::vector<double> next;
    for (const double a : acc) {
      for (const double b : one) {
        next.push_back(a * b);
      }
    }
    acc = std::move(next);
  }
  return acc;
}

/// The relative rounding of one entry of `M E` and of its use, in units of `eps`, for the bound of
/// a sum over non-negative (Bezier) or signed terms of `|C_jc B_c|`.
///
/// An entry of `M` is the product of at most `levels - 1` refinements, each per direction a sum of
/// at most `p + 1` non-negative terms (two operations each); an entry of `E` is built from the
/// `p^2` operations of the knot insertions of each direction; `M E` and the final sum over the
/// reference basis add `n` terms each, and the closed-form basis `4 p` operations per direction.
/// The terms of `M` and of `E` of the Bezier target are non-negative, so these relative errors
/// bound the absolute one in units of `sum |C B|`.
double operator_units(const Space& space) {
  const auto levels = static_cast<double>(space.num_levels());
  const auto d = static_cast<double>(space.dim());
  double p_max = 0.0;
  double n = 1.0;
  for (const int p : space.degrees()) {
    p_max = (std::max)(p_max, static_cast<double>(p));
    n *= static_cast<double>(p) + 1.0;
  }
  const double units_m = levels * d * 2.0 * (p_max + 1.0);
  const double units_e = d * 2.0 * p_max * p_max + 2.0;
  const double units_basis = d * 4.0 * p_max;
  return units_m + units_e + units_basis + 2.0 * n;
}

/// `(sum_c C(j, c) B_c(xi))` for every row `j`, with `sum_c |C(j, c) B_c(xi)|`.
struct RowSums {
  std::vector<double> value;
  std::vector<double> abs_terms;
};

RowSums apply_operator(const NdArray<double>& op, const std::vector<double>& basis) {
  const std::size_t rows = op.shape()[0];
  const std::size_t cols = op.shape()[1];
  REQUIRE(cols == basis.size());
  RowSums out;
  out.value.assign(rows, 0.0);
  out.abs_terms.assign(rows, 0.0);
  for (std::size_t j = 0; j < rows; ++j) {
    for (std::size_t c = 0; c < cols; ++c) {
      const double term = op.data()[j * cols + c] * basis[c];
      out.value[j] += term;
      out.abs_terms[j] += std::abs(term);
    }
  }
  return out;
}

/// `M E` reproduces the space's basis on every cell: at generic reference points of the cell,
/// `sum_c C(j, c) B_c(xi) = H_j(x)`. The error is the tabulation's own, relative to the value
/// (`value_error_units`), plus that of the operator and its use, relative to `sum |C B|`
/// (`operator_units`), times `build_slack`.
void check_reference_basis(const Space& space, Target target) {
  const MultiLevelExtraction<double> mle(space, target);
  const std::size_t d = space.dim();
  const std::vector<int> degrees(space.degrees().begin(), space.degrees().end());
  const double tab_units = value_error_units(space);
  const double op_units = operator_units(space);
  const std::vector<double> ts = {0.21, 0.5, 0.83};
  for (I64 cid = 0; cid < space.grid().num_cells(); ++cid) {
    CAPTURE(cid);
    const NdArray<double> op = mle.extraction_operator(cid);
    const std::vector<double> pts = thb_test::cell_points(space, cid, ts);
    const std::size_t n_pts = pts.size() / d;
    const auto [lo, hi] = space.grid().cell_bounds(cid);
    const auto tab = space.tabulate_basis(cid, mdview<const double, 2>(pts.data(), n_pts, d));
    const std::size_t k = tab.dofs.size();
    REQUIRE(op.shape()[0] == k);
    for (std::size_t i = 0; i < n_pts; ++i) {
      std::vector<double> xi(d);
      for (std::size_t a = 0; a < d; ++a) {
        xi[a] = (pts[i * d + a] - lo[a]) / (hi[a] - lo[a]);
      }
      const RowSums rs = apply_operator(op, basis_nd(target, degrees, xi.data()));
      for (std::size_t j = 0; j < k; ++j) {
        const double h = tab.values.data()[i * k + j];
        const double bound =
            build_slack * eps * (tab_units * std::abs(h) + op_units * rs.abs_terms[j]);
        CHECK(std::abs(rs.value[j] - h) <= bound);
      }
    }
  }
}

TEST_CASE("MultiLevelExtraction: M E maps the Bernstein basis to the space's functions") {
  const auto spaces = test_spaces();
  SUBCASE("2D THB, three levels") { check_reference_basis(spaces[0], Target::bezier); }
  SUBCASE("2D HB, three levels") { check_reference_basis(spaces[1], Target::bezier); }
  SUBCASE("3D THB") { check_reference_basis(spaces[2], Target::bezier); }
  SUBCASE("1D cubic THB, regularity 0") { check_reference_basis(spaces[3], Target::bezier); }
  SUBCASE("2D THB, mixed degrees") { check_reference_basis(spaces[4], Target::bezier); }
  SUBCASE("2D THB, factor 1 along an axis") { check_reference_basis(spaces[5], Target::bezier); }
}

TEST_CASE("MultiLevelExtraction: M E maps the Lagrange basis to the space's functions") {
  const auto spaces = test_spaces();
  SUBCASE("2D THB, three levels") { check_reference_basis(spaces[0], Target::lagrange); }
  SUBCASE("1D cubic THB, regularity 0") { check_reference_basis(spaces[3], Target::lagrange); }
  SUBCASE("2D THB, mixed degrees") { check_reference_basis(spaces[4], Target::lagrange); }
}

TEST_CASE("MultiLevelExtraction: the columns of M E of a truncated space sum to one") {
  const auto spaces = test_spaces();
  for (const Target target : {Target::bezier, Target::lagrange, Target::cardinal}) {
    for (const std::size_t at : {std::size_t{0}, std::size_t{2}, std::size_t{4}}) {
      const Space& space = spaces[at];
      const MultiLevelExtraction<double> mle(space, target);
      const double op_units = operator_units(space);
      for (I64 cid = 0; cid < space.grid().num_cells(); ++cid) {
        const NdArray<double> op = mle.extraction_operator(cid);
        const std::size_t rows = op.shape()[0];
        const std::size_t cols = op.shape()[1];
        for (std::size_t c = 0; c < cols; ++c) {
          double sum = 0.0;
          double abs_sum = 0.0;
          for (std::size_t j = 0; j < rows; ++j) {
            sum += op.data()[j * cols + c];
            abs_sum += std::abs(op.data()[j * cols + c]);
          }
          CHECK(std::abs(sum - 1.0) <= build_slack * eps * (op_units + static_cast<double>(rows)) *
                                           (std::max)(abs_sum, 1.0));
        }
      }
    }
  }
}

TEST_CASE("MultiLevelExtraction: the plain hierarchical basis is not a partition of unity") {
  const Space hb = thb_test::refined_2d(false);
  const MultiLevelExtraction<double> mle(hb, Target::bezier);
  double worst = 0.0;
  for (I64 cid = 0; cid < hb.grid().num_cells(); ++cid) {
    const NdArray<double> op = mle.extraction_operator(cid);
    for (std::size_t c = 0; c < op.shape()[1]; ++c) {
      double sum = 0.0;
      for (std::size_t j = 0; j < op.shape()[0]; ++j) {
        sum += op.data()[j * op.shape()[1] + c];
      }
      worst = (std::max)(worst, std::abs(sum - 1.0));
    }
  }
  // A control threshold needs only to be far above the round-off level (a few `eps`) and far below
  // the effect it detects (the weight of the coarse functions that truncation would have removed,
  // a sizeable fraction of one): 0.05 is both, by orders of magnitude on either side.
  CHECK(worst > 0.05);
}

TEST_CASE("MultiLevelExtraction: M does not depend on the target") {
  const Space space = thb_test::refined_2d(true);
  const MultiLevelExtraction<double> bezier(space, Target::bezier);
  const MultiLevelExtraction<double> cardinal(space, Target::cardinal);
  for (I64 cid = 0; cid < space.grid().num_cells(); ++cid) {
    const NdArray<double> a = bezier.multilevel_operator(cid);
    const NdArray<double> b = cardinal.multilevel_operator(cid);
    REQUIRE(a.size() == b.size());
    CHECK(std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0);
  }
}

TEST_CASE("MultiLevelExtraction: an unrefined space has the identity for M") {
  const auto knots = thb_test::uniform_knots(3);
  const Space space =
      thb_test::make_thb_space({knots, knots}, {2, 2}, {2, 2}, {std::nullopt, std::nullopt}, true);
  const MultiLevelExtraction<double> mle(space, Target::bezier);
  const bs::SpanwiseElementExtraction<double> single(space.level_space(0), Target::bezier);
  CHECK(mle.num_elements() == 9);
  for (I64 cid = 0; cid < 9; ++cid) {
    const NdArray<double> m = mle.multilevel_operator(cid);
    // The functions of a root cell are the level-0 functions supported on it, in dof order, which
    // is the C order of the cell's `3 x 3` window.
    REQUIRE(m.shape()[0] == 9);
    REQUIRE(m.shape()[1] == 9);
    for (std::size_t r = 0; r < 9; ++r) {
      for (std::size_t c = 0; c < 9; ++c) {
        CHECK(m.data()[r * 9 + c] == (r == c ? 1.0 : 0.0));
      }
    }
    // E is the single-level operator of the cell, which `kron_operator` forms.
    const std::vector<I64> midx = space.grid().cell_multi_index(cid);
    const std::vector<std::size_t> cell(midx.begin(), midx.end());
    std::vector<double> e(81);
    single.kron_operator(cell, mdview<double, 2>(e.data(), 9, 9));
    const NdArray<double> c = mle.extraction_operator(cid);
    for (std::size_t i = 0; i < 81; ++i) {
      // An identity times E: each entry is `1 * e` plus exact zeros.
      CHECK(c.data()[i] == e[i]);
    }
  }
}

TEST_CASE(
    "MultiLevelExtraction: every target's single-level operator is square in the cell's width") {
  // `extraction_operator` multiplies `M` (width prod(p + 1)) by `E` without checking that `E` has
  // that many rows: each direction's operator is (p + 1) x (p + 1) for every target.
  const Space space = thb_test::refined_2d(true);
  for (const Target target : {Target::bezier, Target::lagrange, Target::cardinal}) {
    const MultiLevelExtraction<double> mle(space, target);
    for (I64 cid = 0; cid < space.grid().num_cells(); ++cid) {
      CHECK(mle.extraction_operator(cid).shape()[1] == mle.multilevel_operator(cid).shape()[1]);
    }
    for (I64 level = 0; level < space.num_levels(); ++level) {
      const bs::SpanwiseElementExtraction<double> ext(space.level_space(level), target);
      std::size_t in = 1;
      std::size_t out = 1;
      for (const std::size_t v : ext.input_shape_per_dir()) {
        in *= v;
      }
      for (const std::size_t v : ext.output_shape_per_dir()) {
        out *= v;
      }
      CHECK(in == 9);
      CHECK(out == 9);
    }
  }
}

TEST_CASE("MultiLevelExtraction: the recorded Python operators of a three-level space") {
  const Space space = thb_test::refined_2d(true);
  REQUIRE(space.grid().num_cells() == gold::num_cells);
  REQUIRE(space.num_total_basis() == gold::num_total);
  const MultiLevelExtraction<double> mle(space, Target::bezier);
  const double units = operator_units(space);
  for (const gold::Cell& g : gold::cells) {
    CAPTURE(g.level);
    CAPTURE(g.i);
    CAPTURE(g.j);
    const std::optional<I64> cid = space.grid().cell_id(g.level, Ivec{g.i, g.j});
    REQUIRE(cid.has_value());
    CHECK(*cid == g.cid);
    const auto dofs = mle.active_basis(*cid);
    REQUIRE(dofs.size() == g.dofs.size());
    CHECK(std::equal(dofs.begin(), dofs.end(), g.dofs.begin()));
    const NdArray<double> m = mle.multilevel_operator(*cid);
    const NdArray<double> c = mle.extraction_operator(*cid);
    REQUIRE(m.size() == g.m.size());
    REQUIRE(c.size() == g.c.size());
    REQUIRE(m.shape()[0] == static_cast<std::size_t>(g.rows));
    REQUIRE(m.shape()[1] == static_cast<std::size_t>(g.cols));
    // Two computations each carrying `operator_units` of relative error, and a recorded entry that
    // is exactly zero is zero in both (the entries are sums of non-negative terms).
    for (std::size_t i = 0; i < m.size(); ++i) {
      CHECK(std::abs(m.data()[i] - g.m[i]) <= 2.0 * build_slack * eps * units * std::abs(g.m[i]));
    }
    for (std::size_t i = 0; i < c.size(); ++i) {
      CHECK(std::abs(c.data()[i] - g.c[i]) <= 2.0 * build_slack * eps * units * std::abs(g.c[i]));
    }
  }
}

TEST_CASE("MultiLevelExtraction: the refusals and their messages") {
  const Space space = thb_test::refined_2d(true);
  const MultiLevelExtraction<double> mle(space, Target::bezier);
  const I64 n = space.grid().num_cells();
  const std::string range = " is out of range [0, " + std::to_string(n) + ").";
  THB_THROWS_WITH_AS(mle.multilevel_operator(n), ("cell id " + std::to_string(n) + range).c_str(),
                     pantr::index_error);
  THB_THROWS_WITH_AS(mle.extraction_operator(-1), ("cell id -1" + range).c_str(),
                     pantr::index_error);
  THB_THROWS_WITH_AS(mle.active_basis(n), ("cell id " + std::to_string(n) + range).c_str(),
                     pantr::index_error);

  const std::size_t k = mle.active_basis(0).size();
  std::vector<double> wrong(4);
  const std::string want =
      "Output array has shape (2, 2), but expected shape (" + std::to_string(k) + ", 9)";
  THB_THROWS_WITH_AS(mle.multilevel_operator(0, mdview<double, 2>(wrong.data(), 2, 2)),
                     want.c_str(), pantr::value_error);
  THB_THROWS_WITH_AS(mle.extraction_operator(0, mdview<double, 2>(wrong.data(), 2, 2)),
                     want.c_str(), pantr::value_error);
}

TEST_CASE("MultiLevelExtraction: a caller's output array is written in full") {
  const Space space = thb_test::refined_2d(true);
  const MultiLevelExtraction<double> mle(space, Target::bezier);
  for (const I64 cid : {I64{0}, space.grid().num_cells() - 1}) {
    const NdArray<double> reference = mle.extraction_operator(cid);
    // A buffer full of NaN: any entry the call left alone shows.
    std::vector<double> buffer(reference.size(), std::numeric_limits<double>::quiet_NaN());
    mle.extraction_operator(
        cid, mdview<double, 2>(buffer.data(), reference.shape()[0], reference.shape()[1]));
    CHECK(std::memcmp(buffer.data(), reference.data(), buffer.size() * sizeof(double)) == 0);
  }
}

TEST_CASE("MultiLevelExtraction: a float root on dyadic knots gives the double root's operators") {
  // Oracle: the double root's extraction of the same space. The knots are exact in float, so the
  // two spaces have the same cells and the same active functions (integers, exactly equal). Every
  // two-scale entry and every single-level extraction entry of a degree-2 space on dyadic spans is
  // a ratio of knot differences that are powers of two (the spans 1/4, 1/2 and, after refinement,
  // 1/8, 1/16), so the float arithmetic is exact and `M` and `M E` agree bit for bit.
  const THBSplineSpace<float> sf = thb_test::refined_2d<float>(true);
  const Space sd = thb_test::refined_2d<double>(true);
  REQUIRE(sf.grid().num_cells() == sd.grid().num_cells());
  const MultiLevelExtraction<float> mf(sf, Target::bezier);
  const MultiLevelExtraction<double> md(sd, Target::bezier);
  for (I64 cid = 0; cid < sd.grid().num_cells(); ++cid) {
    CAPTURE(cid);
    const std::span<const I64> rows_f = mf.active_basis(cid);
    const std::span<const I64> rows_d = md.active_basis(cid);
    CHECK(std::equal(rows_f.begin(), rows_f.end(), rows_d.begin(), rows_d.end()));
    const NdArray<double> m_f = mf.multilevel_operator(cid);
    const NdArray<double> m_d = md.multilevel_operator(cid);
    REQUIRE(m_f.size() == m_d.size());
    CHECK(std::memcmp(m_f.data(), m_d.data(), m_f.size() * sizeof(double)) == 0);
    const NdArray<double> e_f = mf.extraction_operator(cid);
    const NdArray<double> e_d = md.extraction_operator(cid);
    REQUIRE(e_f.size() == e_d.size());
    CHECK(std::memcmp(e_f.data(), e_d.data(), e_f.size() * sizeof(double)) == 0);
  }
}

TEST_CASE(
    "MultiLevelExtraction: a float root with a ratio that is not dyadic agrees to float rounding") {
  // A cubic on four spans: the knot differences of three spans (3/4) are not powers of two, so the
  // float two-scale entries are rounded. Oracle: the double root's operators. An entry of `M` is a
  // sum of non-negative products of `levels - 1` refinements, each direction contributing a
  // two-scale entry of relative error `6 p + 1` roundings (the count the B-spline product test
  // derives for an Oslo row entry); `M E` adds the single-level operator's entry, of the same
  // count. All entries lie in `[0, 1]` (a column of `M E` is a partition of unity of non-negative
  // terms), so the absolute error is bounded by the relative one, times `build_slack` and the unit
  // roundoff of float.
  const auto make = [](auto tag) {
    using T = decltype(tag);
    const auto root = thb_test::make_thb_space<T>({k3}, {3}, {2}, {0}, true);
    return root.refine_region(0, Ivec{0}, Ivec{2}, std::nullopt)
        .refine_region(1, Ivec{0}, Ivec{2}, std::nullopt);
  };
  const THBSplineSpace<float> sf = make(float{});
  const Space sd = make(double{});
  REQUIRE(sf.grid().num_cells() == sd.grid().num_cells());
  const double u_f = static_cast<double>(std::numeric_limits<float>::epsilon()) / 2.0;
  const double steps = static_cast<double>(sd.num_levels() - 1);
  const double per_step = 6.0 * sd.degrees()[0] + 1.0;
  const double bound_m = build_slack * steps * per_step * u_f;
  const double bound_me = build_slack * (steps + 1.0) * per_step * u_f;
  CHECK(bound_me < 1e-4);  // not vacuous: the entries are of order one
  const MultiLevelExtraction<float> mf(sf, Target::bezier);
  const MultiLevelExtraction<double> md(sd, Target::bezier);
  for (I64 cid = 0; cid < sd.grid().num_cells(); ++cid) {
    CAPTURE(cid);
    const NdArray<double> m_f = mf.multilevel_operator(cid);
    const NdArray<double> m_d = md.multilevel_operator(cid);
    const NdArray<double> e_f = mf.extraction_operator(cid);
    const NdArray<double> e_d = md.extraction_operator(cid);
    REQUIRE(m_f.size() == m_d.size());
    REQUIRE(e_f.size() == e_d.size());
    for (std::size_t i = 0; i < m_f.size(); ++i) {
      CHECK(std::abs(m_f.data()[i] - m_d.data()[i]) <= bound_m);
    }
    for (std::size_t i = 0; i < e_f.size(); ++i) {
      CHECK(std::abs(e_f.data()[i] - e_d.data()[i]) <= bound_me);
    }
  }
}

TEST_CASE("MultiLevelExtraction: threads racing on the first operator agree with a serial build") {
  // A TSan target: the refined boxes are one lazy block, computed once under the block's lock (in
  // parallel over functions inside it), and every thread then reads the same block. The reference
  // is built with one thread, so the comparison also covers the thread count.
  constexpr int n_threads = 8;
  const Space space = thb_test::refined_2d(true);
  const auto n_cells = static_cast<std::size_t>(space.grid().num_cells());
  std::vector<NdArray<double>> expected;
  {
    const thb_test::ThreadsGuard serial(1);
    const MultiLevelExtraction<double> reference(space, Target::bezier);
    for (std::size_t c = 0; c < n_cells; ++c) {
      expected.push_back(reference.multilevel_operator(static_cast<I64>(c)));
    }
  }
  for (int round = 0; round < 10; ++round) {
    const MultiLevelExtraction<double> mle(space, Target::bezier);
    std::atomic<int> ready{0};
    std::vector<std::size_t> mismatches(n_threads, 0);
    std::vector<std::thread> threads;
    for (int t = 0; t < n_threads; ++t) {
      threads.emplace_back([&, t] {
        ready.fetch_add(1);
        while (ready.load() < n_threads) {
        }
        for (std::size_t c = 0; c < n_cells; ++c) {
          const std::size_t j = (c + static_cast<std::size_t>(t) * 5) % n_cells;
          const NdArray<double> got = mle.multilevel_operator(static_cast<I64>(j));
          if (got.size() != expected[j].size() ||
              std::memcmp(got.data(), expected[j].data(), got.size() * sizeof(double)) != 0) {
            ++mismatches[static_cast<std::size_t>(t)];
          }
        }
      });
    }
    for (auto& th : threads) {
      th.join();
    }
    for (int t = 0; t < n_threads; ++t) {
      CHECK(mismatches[static_cast<std::size_t>(t)] == 0);
    }
  }
}

TEST_CASE("MultiLevelExtraction: a copy shares the computed boxes and agrees") {
  const Space space = thb_test::refined_2d(true);
  const MultiLevelExtraction<double> a(space, Target::bezier);
  const NdArray<double> first = a.multilevel_operator(0);
  const MultiLevelExtraction<double> b = a;  // NOLINT(performance-unnecessary-copy-initialization)
  const NdArray<double> second = b.multilevel_operator(0);
  REQUIRE(first.size() == second.size());
  CHECK(std::memcmp(first.data(), second.data(), first.size() * sizeof(double)) == 0);
}

}  // namespace
