// What the tests of SpanwiseElementExtraction and of the Kronecker application behind it share:
// dense reference matrices in exact rational arithmetic, the derived error bound of a contraction
// (see the oracle note in test_kron_apply.cpp), deterministic data, and the builders of the test
// spaces and of the reference operators.
#pragma once

#include "bspline_oracle.hpp"
#include "bspline_test_support.hpp"

#include <doctest/doctest.h>

#include <pantr/basis/lagrange_variant.hpp>
#include <pantr/bspline/extraction.hpp>
#include <pantr/bspline/space.hpp>
#include <pantr/bspline/spanwise_extraction.hpp>
#include <pantr/core/ndarray.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace spanwise_test {

namespace bs = pantr::bspline;
namespace oracle = bspline_oracle;
using bspline_test::Dir;
using oracle::Rational;
using pantr::basis::LagrangeVariant;

/// A dense matrix with its entrywise absolute values (in double, for the error bound) and,
/// unless `exact` is false, its exact values. A matrix without exact values is only a bound.
struct Dense {
  std::size_t rows = 0;
  std::size_t cols = 0;
  bool exact = true;
  std::vector<Rational> v;
  std::vector<double> a;
};

inline Dense make_dense(std::size_t rows, std::size_t cols, bool exact) {
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

inline Dense identity_dense(std::size_t n, bool exact = true) {
  Dense m = make_dense(n, n, exact);
  for (std::size_t i = 0; i < n; ++i) {
    if (exact) {
      m.v[i * n + i] = 1;
    }
    m.a[i * n + i] = 1.0;
  }
  return m;
}

inline Dense transpose(const Dense& m) {
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

inline Dense multiply(const Dense& x, const Dense& y) {
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
inline Dense kron2(const Dense& x, const Dense& y) {
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

inline Dense kron_all(const std::vector<Dense>& factors) {
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

inline double gamma_of(std::size_t n, double u) {
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

/// The reference of an operation on the Kronecker product of `factors`.
struct Reference {
  Dense result;
  std::size_t terms;  // the sum of the contracted lengths
};

/// The exact result of `op` on `operand` (a column vector `(n, 1)` or a square matrix), and the
/// number of terms of its stages.
inline Reference reference_of(bs::ExtractionOp op, const std::vector<Dense>& factors,
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

inline constexpr bs::ExtractionOp kAllOps[] = {bs::ExtractionOp::apply,
                                               bs::ExtractionOp::apply_transpose,
                                               bs::ExtractionOp::mt_k_m, bs::ExtractionOp::m_k_mt};

inline bool is_vector_op(bs::ExtractionOp op) {
  return op == bs::ExtractionOp::apply || op == bs::ExtractionOp::apply_transpose;
}

inline constexpr bs::ExtractionTarget kAllTargets[] = {
    bs::ExtractionTarget::bezier, bs::ExtractionTarget::lagrange, bs::ExtractionTarget::cardinal};

/// A direction with a mix of identity and non-identity elements: the first element is a Bézier
/// patch (multiplicity `degree + 1` at both its ends), the next ones are smooth, the interior of
/// the later ones is uniform.
inline Dir mixed_dir(int degree, int uniform_elements) {
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
inline std::size_t isqrt(std::size_t n) {
  std::size_t r = 0;
  while (r * r < n) {
    ++r;
  }
  REQUIRE(r * r == n);
  return r;
}

inline std::vector<std::size_t> unravel(const std::vector<std::size_t>& extents, std::size_t flat) {
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
inline std::vector<Dir> dirs_for(std::size_t d) {
  const int degrees[] = {2, 3, 1, 2};
  const int uniform[] = {3, 2, 3, 2};
  std::vector<Dir> dirs;
  for (std::size_t k = 0; k < d; ++k) {
    const int degree = d == 4 ? (k == 1 ? 2 : 1) : degrees[k];
    dirs.push_back(mixed_dir(degree, uniform[k]));
  }
  return dirs;
}

}  // namespace spanwise_test
