/// @file product.hpp
/// @brief The exact pointwise product of two B-splines (`multiply`), in any parametric dimension.
///
/// Both operands are refined to Bézier form, the Bernstein product formula is applied element by
/// element, and the result is re-expressed on the **minimal** product knot vectors. Per direction
/// the product has degree `r = p + q`, and at a breakpoint `xi` the multiplicity
///
///     m_h(xi) = max{ m_f(xi) + q : xi a knot of f } ∪ { m_g(xi) + p : xi a knot of g },
///
/// the maximum over the operands that carry a knot there: `h = f g` is `C^min(s_f, s_g)` at `xi`
/// with `s = degree - multiplicity`, and the Curry-Schoenberg theorem (Curry and Schoenberg,
/// *J. Analyse Math.* 17, 1966, 71-107; Lyche and Mørken, *Spline Methods*, Thm 3.25) identifies
/// the spline space with the piecewise polynomials of that smoothness, so the multiplicity is
/// minimal, not merely sufficient. A `C^-1` operand (`m = p + 1`) is inside the range.
///
/// Rational operands are multiplied in homogeneous form: numerators and weights separately, a
/// non-rational operand taking unit weights. Each direction's boundary structure is kept: either
/// operand open gives an open direction, both periodic a periodic one, otherwise unclamped. The
/// product is formed in open form and a non-open direction is converted at the end through its
/// ghost knot vector and a least-squares change of representation.
///
/// The arithmetic follows the Python implementation this replaces step by step, including where it
/// rounded in the storage type `T` (knot arithmetic on `float32` arrays) and where in `double`
/// (Python floats, and `numpy.linalg.lstsq`, which decomposes in `double`;
/// `linalg::lstsq_min_norm`). The one-direction and the multi-direction paths of that
/// implementation differ in two tolerances (which knots count as one breakpoint of the product);
/// both are kept, named where they apply.
#pragma once

#include "pantr/bezier/product.hpp"
#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/detail/refine_common.hpp"
#include "pantr/bspline/knot_insertion.hpp"
#include "pantr/bspline/knots.hpp"
#include "pantr/bspline/space.hpp"
#include "pantr/bspline/space_1d.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/format.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"
#include "pantr/linalg/dense.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace pantr::bspline {

namespace detail {

/// The boundary structure one direction of a product keeps.
enum class ProductBoundary : std::uint8_t {
  open,      ///< Either operand is open (clamped, non-periodic).
  periodic,  ///< Both operands are periodic.
  nonopen,   ///< Otherwise: unclamped.
};

/// The interior breakpoints of a space and their multiplicities.
///
/// Attributes:
///     knots (std::vector<T>): Interior breakpoints, ascending.
///     mults (std::vector<std::int64_t>): Their multiplicities.
template <StorageFloat T>
struct Breakpoints {
  std::vector<T> knots;
  std::vector<std::int64_t> mults;
};

/// Strip the two domain-end classes off the in-domain knot classes.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat T>
[[nodiscard]] Breakpoints<T> interior_classes(std::span<const T> knots, int degree, double tol) {
  KnotClasses<T> c = unique_knots_and_multiplicity<T>(knots, degree, tol, true);
  Breakpoints<T> out;
  if (c.knots.size() > 2) {
    out.knots.assign(c.knots.begin() + 1, c.knots.end() - 1);
    out.mults.assign(c.multiplicities.begin() + 1, c.multiplicities.end() - 1);
  }
  return out;
}

/// Merge two ascending breakpoint lists into their union; two within `tol` become their midpoint.
///
/// The sum of the pair is rounded in `T` and halved in `double`, as the Python did (`float(a + b)
/// / 2.0` on two `T` scalars); the result is stored in `T`.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat T>
[[nodiscard]] std::vector<T> merge_breakpoints(std::span<const T> f, std::span<const T> g,
                                               double tol) {
  std::vector<T> out;
  out.reserve(f.size() + g.size());
  std::size_t i = 0;
  std::size_t j = 0;
  while (i < f.size() && j < g.size()) {
    const auto a = static_cast<double>(f[i]);
    const auto b = static_cast<double>(g[j]);
    if (std::abs(a - b) <= tol) {
      const T sum = f[i] + g[j];
      out.push_back(static_cast<T>(static_cast<double>(sum) / 2.0));
      ++i;
      ++j;
    } else if (a < b) {
      out.push_back(f[i++]);
    } else {
      out.push_back(g[j++]);
    }
  }
  out.insert(out.end(), f.begin() + static_cast<std::ptrdiff_t>(i), f.end());
  out.insert(out.end(), g.begin() + static_cast<std::ptrdiff_t>(j), g.end());
  return out;
}

/// The multiplicity of each merged breakpoint in one operand's list (0 where absent).
///
/// Reproduces the Python lookup exactly: `numpy.searchsorted` (left) of each merged value in the
/// operand's list, then a match only against the entry found there, the difference taken and
/// compared in `T` (a `T` array against a Python float compares in `T`, NEP 50).
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat T>
[[nodiscard]] std::vector<std::int64_t> lookup_multiplicities(std::span<const T> merged,
                                                              const Breakpoints<T>& own,
                                                              double tol) {
  std::vector<std::int64_t> out(merged.size(), 0);
  if (own.knots.empty()) {
    return out;
  }
  const T tol_t = static_cast<T>(tol);
  for (std::size_t k = 0; k < merged.size(); ++k) {
    const auto it = std::lower_bound(own.knots.begin(), own.knots.end(), merged[k]);
    if (it == own.knots.end()) {
      continue;
    }
    const T diff = *it - merged[k];
    if (std::abs(diff) <= tol_t) {
      out[k] = own.mults[static_cast<std::size_t>(it - own.knots.begin())];
    }
  }
  return out;
}

/// The breakpoint structure of one direction shared by two operands and their product.
///
/// Attributes:
///     breakpoints (std::vector<T>): Union of the interior breakpoints, ascending.
///     mults_f (std::vector<std::int64_t>): Multiplicity in the first operand (0 if absent).
///     mults_g (std::vector<std::int64_t>): Multiplicity in the second operand.
///     product (std::vector<std::int64_t>): Minimal multiplicity in the product, in `[1, p+q+1]`.
template <StorageFloat T>
struct ProductMesh {
  std::vector<T> breakpoints;
  std::vector<std::int64_t> mults_f;
  std::vector<std::int64_t> mults_g;
  std::vector<std::int64_t> product;
};

/// Build the product mesh of two 1D spaces of degrees `p` and `q`.
///
/// `tol` groups knots; the breakpoint classes of each space are taken at `tol` rounded to `T`, the
/// merge and the lookup at `tol` itself, as in the Python.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat T>
[[nodiscard]] ProductMesh<T> product_mesh(const BsplineSpace1D<T>& f, const BsplineSpace1D<T>& g,
                                          double tol) {
  const auto tol_typed = static_cast<double>(static_cast<T>(tol));
  const Breakpoints<T> bf = interior_classes<T>(f.knots(), f.degree(), tol_typed);
  const Breakpoints<T> bg = interior_classes<T>(g.knots(), g.degree(), tol_typed);
  ProductMesh<T> m;
  m.breakpoints = merge_breakpoints<T>(bf.knots, bg.knots, tol);
  m.mults_f = lookup_multiplicities<T>(m.breakpoints, bf, tol);
  m.mults_g = lookup_multiplicities<T>(m.breakpoints, bg, tol);
  m.product.resize(m.breakpoints.size());
  const std::int64_t p = f.degree();
  const std::int64_t q = g.degree();
  for (std::size_t k = 0; k < m.breakpoints.size(); ++k) {
    const std::int64_t from_f = m.mults_f[k] > 0 ? m.mults_f[k] + q : 0;
    const std::int64_t from_g = m.mults_g[k] > 0 ? m.mults_g[k] + p : 0;
    m.product[k] = (std::max)(from_f, from_g);
  }
  return m;
}

/// Raise interior multiplicities to Bézier form (`degree` each) without lowering a `C^-1` one.
[[nodiscard]] inline std::vector<std::int64_t> bezier_multiplicities(
    std::int64_t degree, std::span<const std::int64_t> mults) {
  std::vector<std::int64_t> out(mults.size());
  for (std::size_t k = 0; k < mults.size(); ++k) {
    out[k] = (std::max)(degree, mults[k]);
  }
  return out;
}

/// Index of each element's first control point in a Bézier-form direction: `0` then the running
/// sum of the Bézier multiplicities (a `C^-1` breakpoint keeps `degree + 1` and its two elements
/// share no control point).
[[nodiscard]] inline std::vector<std::size_t> element_offsets(
    std::span<const std::int64_t> bezier_mults) {
  std::vector<std::size_t> out(bezier_mults.size() + 1, 0);
  for (std::size_t k = 0; k < bezier_mults.size(); ++k) {
    out[k + 1] = out[k] + static_cast<std::size_t>(bezier_mults[k]);
  }
  return out;
}

/// The knots to insert to bring a space with multiplicities `mults` at `breakpoints` to Bézier
/// form.
template <StorageFloat T>
[[nodiscard]] std::vector<T> knots_for_bezier_form(std::int64_t degree,
                                                   std::span<const T> breakpoints,
                                                   std::span<const std::int64_t> mults) {
  std::vector<T> out;
  for (std::size_t k = 0; k < breakpoints.size(); ++k) {
    const std::int64_t deficit = (std::max)(degree, mults[k]) - mults[k];
    out.insert(out.end(), static_cast<std::size_t>(deficit), breakpoints[k]);
  }
  return out;
}

/// A clamped knot vector on `[a, b]` with `degree + 1` end copies and the given interior ones.
template <StorageFloat T>
[[nodiscard]] std::vector<T> clamped_product_knots(T a, T b, std::span<const T> breakpoints,
                                                   std::span<const std::int64_t> mults,
                                                   std::int64_t degree) {
  std::vector<T> out(static_cast<std::size_t>(degree + 1), a);
  for (std::size_t k = 0; k < breakpoints.size(); ++k) {
    out.insert(out.end(), static_cast<std::size_t>(mults[k]), breakpoints[k]);
  }
  out.insert(out.end(), static_cast<std::size_t>(degree + 1), b);
  return out;
}

/// The multiplicities of the two domain ends among the end runs of a space, counted at `tol`.
///
/// The differences and the comparison are in `T`, as the Python's (`T` array minus a Python
/// float, compared with a Python float, NEP 50).
template <StorageFloat T>
[[nodiscard]] std::pair<std::int64_t, std::int64_t> boundary_multiplicities(
    const BsplineSpace1D<T>& s, double tol) {
  const std::span<const T> k = s.knots();
  const auto p = static_cast<std::size_t>(s.degree());
  const T a = k[p];
  const T b = k[k.size() - p - 1];
  const T tol_t = static_cast<T>(tol);
  std::int64_t left = 0;
  std::int64_t right = 0;
  for (std::size_t i = 0; i <= p; ++i) {
    const T dl = k[i] - a;
    const T dr = k[k.size() - p - 1 + i] - b;
    left += std::abs(dl) <= tol_t ? 1 : 0;
    right += std::abs(dr) <= tol_t ? 1 : 0;
  }
  return {left, right};
}

/// A periodic (ghost-extended) knot vector with the given interior breakpoints.
///
/// One period is the tile `[a^{m_bdy}, xi_1^{m_1}, ...]`; `degree + 1 - m_bdy` ghost knots on
/// each side continue it, shifted by multiples of the period `b - a` (a `double`). Each shifted
/// knot is the tile entry plus the shift rounded to `T`, summed in `T`, as the Python's `T` array
/// plus a Python float.
template <StorageFloat T>
[[nodiscard]] std::vector<T> ghost_product_knots(T a, T b, std::span<const T> breakpoints,
                                                 std::span<const std::int64_t> mults,
                                                 std::int64_t m_bdy, std::int64_t degree) {
  const double period = static_cast<double>(b) - static_cast<double>(a);
  std::vector<T> tile(static_cast<std::size_t>(m_bdy), a);
  for (std::size_t k = 0; k < breakpoints.size(); ++k) {
    tile.insert(tile.end(), static_cast<std::size_t>(mults[k]), breakpoints[k]);
  }
  const std::span<const T> interior =
      std::span<const T>(tile).subspan(static_cast<std::size_t>(m_bdy));
  const auto shifted = [&](std::span<const T> src, long long s, std::vector<T>& dst) {
    const T shift = static_cast<T>(static_cast<double>(s) * period);
    for (const T v : src) {
      const T sum = v + shift;
      dst.push_back(sum);
    }
  };
  const std::int64_t n_ghost = degree + 1 - m_bdy;
  std::vector<T> out;
  if (n_ghost > 0) {
    const auto len = static_cast<std::int64_t>((std::max)(tile.size(), std::size_t{1}));
    const long long n_tiles = (n_ghost + len - 1) / len + 2;
    std::vector<T> left;
    for (long long s = -n_tiles; s < 0; ++s) {
      shifted(tile, s, left);
    }
    out.assign(left.end() - n_ghost, left.end());
  }
  out.insert(out.end(), static_cast<std::size_t>(m_bdy), a);
  for (std::size_t k = 0; k < breakpoints.size(); ++k) {
    out.insert(out.end(), static_cast<std::size_t>(mults[k]), breakpoints[k]);
  }
  out.insert(out.end(), static_cast<std::size_t>(m_bdy), b);
  if (n_ghost > 0) {
    const auto len = static_cast<std::int64_t>((std::max)(tile.size(), std::size_t{1}));
    const long long n_tiles = (n_ghost + len - 1) / len + 2;
    std::vector<T> right;
    shifted(interior, 1, right);
    for (long long s = 2; s < n_tiles + 2; ++s) {
      shifted(tile, s, right);
    }
    out.insert(out.end(), right.begin(), right.begin() + n_ghost);
  }
  return out;
}

/// The ghost knot vector of one product direction and the matrix mapping control points over it
/// to control points over the open product knots `open_knots` (the Oslo chain: ghost to
/// ghost-with-clamped-ends, trimmed, then to the open knots).
///
/// Args:
///     open_knots (std::span<const T>): The open product knot vector of the direction.
///     degree (int): The product degree.
///     m_bdy (std::int64_t): Boundary multiplicity of the ghost knot vector.
///     class_tol (double): Tolerance at which the open knots are grouped into breakpoints.
///
/// Returns:
///     std::pair<std::vector<T>, linalg::RowMatrix<T>>: The ghost knot vector and the matrix,
///     extents `(n_open, n_ghost_full)`, multiplied in `T`.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat T>
[[nodiscard]] std::pair<std::vector<T>, linalg::RowMatrix<T>> ghost_to_open(
    std::span<const T> open_knots, int degree, std::int64_t m_bdy, double class_tol) {
  using Mat = linalg::RowMatrix<T>;
  const auto r = static_cast<std::size_t>(degree);
  const Breakpoints<T> inner = interior_classes<T>(open_knots, degree, class_tol);
  const T a = open_knots[r];
  const T b = open_knots[open_knots.size() - r - 1];
  std::vector<T> ghost = ghost_product_knots<T>(a, b, inner.knots, inner.mults, m_bdy, degree);

  const auto n_ghost = static_cast<std::size_t>(static_cast<std::int64_t>(r) + 1 - m_bdy);
  const T ga = ghost[r];
  const T gb = ghost[ghost.size() - r - 1];
  std::vector<T> inter = ghost;
  inter.insert(inter.end(), n_ghost, ga);
  inter.insert(inter.end(), n_ghost, gb);
  std::sort(inter.begin(), inter.end());
  const NdArray<T> oslo1 = oslo_matrix<T>(degree, ghost, inter);

  const std::span<const T> per_open =
      std::span<const T>(inter).subspan(n_ghost, inter.size() - 2 * n_ghost);
  const std::size_t n_per_open = per_open.size() - r - 1;
  const NdArray<T> oslo2 = oslo_matrix<T>(degree, per_open, open_knots);
  const Mat trimmed = linalg::as_matrix(oslo1).middleRows(static_cast<Eigen::Index>(n_ghost),
                                                          static_cast<Eigen::Index>(n_per_open));
  Mat chain = linalg::as_matrix(oslo2) * trimmed;
  return {std::move(ghost), std::move(chain)};
}

/// Convert one direction's flattened open product control points to its target boundary form.
///
/// Args:
///     open_space (const BsplineSpace1D<T>&): The open product space of the direction.
///     flat (mdview<const T, 2>): Control points flattened along the direction, `(n_open, rest)`.
///     target (ProductBoundary): `periodic` or `nonopen`.
///     f (const BsplineSpace1D<T>&): The first operand's original space in the direction.
///     g (const BsplineSpace1D<T>&): The second operand's original space.
///     one_direction (bool): Whether the product has one parametric direction. The Python's
///         one-direction path grouped the open product knots at the product space's own tolerance,
///         its multi-direction path at the operands' larger tolerance rounded to `T`.
///
/// Returns:
///     std::pair<BsplineSpace1D<T>, NdArray<T>>: The converted space and flattened points.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat T>
[[nodiscard]] std::pair<BsplineSpace1D<T>, NdArray<T>> convert_product_boundary(
    const BsplineSpace1D<T>& open_space, mdview<const T, 2> flat, ProductBoundary target,
    const BsplineSpace1D<T>& f, const BsplineSpace1D<T>& g, bool one_direction) {
  using Mat = linalg::RowMatrix<T>;
  const std::int64_t p = f.degree();
  const std::int64_t q = g.degree();
  const int r = open_space.degree();
  const double tol = (std::max)(f.tolerance(), g.tolerance());
  const auto [mf_l, mf_r] = boundary_multiplicities<T>(f, tol);
  const auto [mg_l, mg_r] = boundary_multiplicities<T>(g, tol);
  const std::int64_t m_bdy =
      target == ProductBoundary::periodic
          ? (std::max)(mf_l + q, mg_l + p)
          : (std::min)((std::max)(mf_l + q, mg_l + p), (std::max)(mf_r + q, mg_r + p));
  const auto tol_typed = static_cast<double>(static_cast<T>(tol));
  const double class_tol = one_direction ? open_space.tolerance() : tol_typed;
  auto [ghost, chain] = ghost_to_open<T>(open_space.knots(), r, m_bdy, class_tol);

  const Mat rhs =
      Eigen::Map<const Mat>(flat.data_handle(), static_cast<Eigen::Index>(flat.extent(0)),
                            static_cast<Eigen::Index>(flat.extent(1)));
  if (target == ProductBoundary::periodic) {
    const auto n_full = static_cast<Eigen::Index>(ghost.size()) - r - 1;
    const auto n_per = static_cast<Eigen::Index>(num_basis<T>(ghost, r, true, tol_typed));
    Mat wrap = Mat::Zero(n_full, n_per);
    for (Eigen::Index i = 0; i < n_full; ++i) {
      wrap(i, i % n_per) = T(1);
    }
    const Mat system = chain * wrap;
    NdArray<T> ctrl = linalg::lstsq_min_norm<T>(system, rhs);
    return {BsplineSpace1D<T>(std::span<const T>(ghost), r, true), std::move(ctrl)};
  }
  NdArray<T> ctrl = linalg::lstsq_min_norm<T>(chain, rhs);
  return {BsplineSpace1D<T>(std::span<const T>(ghost), r), std::move(ctrl)};
}

/// Copy the box `[start_d, start_d + count_d)` of a row-major array of extents `shape` (the last
/// extent, the coordinates, taken whole) into a contiguous buffer, or write one back.
template <StorageFloat T>
void copy_box(const T* src, std::span<const std::size_t> src_shape,
              std::span<const std::size_t> start, std::span<const std::size_t> count, T* dst,
              std::span<const std::size_t> dst_shape, std::span<const std::size_t> dst_start) {
  const std::size_t dim = count.size();
  const std::size_t cp = src_shape.back();
  std::vector<std::size_t> src_stride(dim + 1, cp);
  std::vector<std::size_t> dst_stride(dim + 1, cp);
  for (std::size_t d = dim; d-- > 0;) {
    src_stride[d] = src_stride[d + 1] * (d + 1 < dim ? src_shape[d + 1] : 1);
    dst_stride[d] = dst_stride[d + 1] * (d + 1 < dim ? dst_shape[d + 1] : 1);
  }
  std::vector<std::size_t> idx(dim, 0);
  while (true) {
    std::size_t s = 0;
    std::size_t t = 0;
    for (std::size_t d = 0; d < dim; ++d) {
      s += (start[d] + idx[d]) * src_stride[d];
      t += (dst_start[d] + idx[d]) * dst_stride[d];
    }
    std::copy_n(src + s, cp, dst + t);
    std::size_t d = dim;
    while (d > 0) {
      --d;
      if (++idx[d] < count[d]) {
        break;
      }
      idx[d] = 0;
      if (d == 0) {
        return;
      }
    }
    if (dim == 0) {
      return;
    }
  }
}

/// Multiply two non-rational B-splines that are open in every direction, on the minimal product
/// knot vectors.
///
/// The Bézier-form product is projected onto the minimal knot vectors in **every** direction when
/// any direction needs it (an exact change of representation by least squares, the minimal knot
/// vector being a subsequence of the Bézier one), as the Python did.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
template <StorageFloat T>
[[nodiscard]] Bspline<T> multiply_open_nonrational(const Bspline<T>& f0, const Bspline<T>& g0) {
  const BsplineSpace<T> space_f = f0.space();
  const BsplineSpace<T> space_g = g0.space();
  const std::vector<BsplineSpace1D<T>>& sf = space_f.spaces();
  const std::vector<BsplineSpace1D<T>>& sg = space_g.spaces();
  const std::size_t dim = sf.size();
  double tol = 0.0;
  for (std::size_t d = 0; d < dim; ++d) {
    tol = (std::max)({tol, sf[d].tolerance(), sg[d].tolerance()});
  }

  std::vector<ProductMesh<T>> meshes;
  std::vector<std::vector<std::int64_t>> bez_h(dim);
  std::vector<std::vector<std::size_t>> off_f(dim);
  std::vector<std::vector<std::size_t>> off_g(dim);
  std::vector<std::vector<std::size_t>> off_h(dim);
  std::vector<std::vector<T>> ins_f(dim);
  std::vector<std::vector<T>> ins_g(dim);
  std::vector<std::int64_t> deg_h(dim);
  bool any_f = false;
  bool any_g = false;
  for (std::size_t d = 0; d < dim; ++d) {
    meshes.push_back(product_mesh<T>(sf[d], sg[d], tol));
    const ProductMesh<T>& m = meshes.back();
    const std::int64_t p = sf[d].degree();
    const std::int64_t q = sg[d].degree();
    deg_h[d] = p + q;
    bez_h[d] = bezier_multiplicities(deg_h[d], m.product);
    off_f[d] = element_offsets(bezier_multiplicities(p, m.mults_f));
    off_g[d] = element_offsets(bezier_multiplicities(q, m.mults_g));
    off_h[d] = element_offsets(bez_h[d]);
    ins_f[d] = knots_for_bezier_form<T>(p, m.breakpoints, m.mults_f);
    ins_g[d] = knots_for_bezier_form<T>(q, m.breakpoints, m.mults_g);
    any_f = any_f || !ins_f[d].empty();
    any_g = any_g || !ins_g[d].empty();
  }
  const auto refine = [dim](const Bspline<T>& b, const std::vector<std::vector<T>>& ins) {
    std::vector<std::span<const T>> spans(dim);
    for (std::size_t d = 0; d < dim; ++d) {
      spans[d] = ins[d];
    }
    return insert_knots<T>(b, spans);
  };
  const Bspline<T> f = any_f ? refine(f0, ins_f) : f0;
  const Bspline<T> g = any_g ? refine(g0, ins_g) : g0;

  // Element by element, in row-major element order; a control point two elements share is written
  // twice with the same value (the later write stands, as in the Python).
  const std::size_t cp = f.cp_size();
  std::vector<std::size_t> shape_h(dim + 1, cp);
  std::vector<std::size_t> n_elem(dim);
  for (std::size_t d = 0; d < dim; ++d) {
    shape_h[d] = off_h[d].back() + static_cast<std::size_t>(deg_h[d]) + 1;
    n_elem[d] = meshes[d].breakpoints.size() + 1;
  }
  NdArray<T> ctrl_h(shape_h, for_overwrite);
  std::vector<std::size_t> pf(dim + 1, cp);
  std::vector<std::size_t> pg(dim + 1, cp);
  std::vector<std::size_t> cnt_f(dim);
  std::vector<std::size_t> cnt_g(dim);
  std::vector<std::size_t> cnt_h(dim);
  std::vector<std::size_t> zeros(dim, 0);
  std::size_t size_f = cp;
  std::size_t size_g = cp;
  for (std::size_t d = 0; d < dim; ++d) {
    cnt_f[d] = static_cast<std::size_t>(sf[d].degree()) + 1;
    cnt_g[d] = static_cast<std::size_t>(sg[d].degree()) + 1;
    cnt_h[d] = static_cast<std::size_t>(deg_h[d]) + 1;
    pf[d] = cnt_f[d];
    pg[d] = cnt_g[d];
    size_f *= cnt_f[d];
    size_g *= cnt_g[d];
  }
  std::vector<T> patch_f(size_f);
  std::vector<T> patch_g(size_g);
  std::vector<std::size_t> e(dim, 0);
  std::vector<std::size_t> st_f(dim);
  std::vector<std::size_t> st_g(dim);
  std::vector<std::size_t> st_h(dim);
  while (true) {
    for (std::size_t d = 0; d < dim; ++d) {
      st_f[d] = off_f[d][e[d]];
      st_g[d] = off_g[d][e[d]];
      st_h[d] = off_h[d][e[d]];
    }
    copy_box<T>(f.control_points().data(), f.shape(), st_f, cnt_f, patch_f.data(), pf, zeros);
    copy_box<T>(g.control_points().data(), g.shape(), st_g, cnt_g, patch_g.data(), pg, zeros);
    const NdArray<T> prod = bezier::bernstein_product<T>(patch_f, pf, patch_g, pg);
    copy_box<T>(prod.data(), prod.shape(), zeros, cnt_h, ctrl_h.data(), shape_h, st_h);
    std::size_t d = dim;
    bool done = true;
    while (d > 0) {
      --d;
      if (++e[d] < n_elem[d]) {
        done = false;
        break;
      }
      e[d] = 0;
    }
    if (done) {
      break;
    }
  }

  std::vector<std::vector<T>> knots_full(dim);
  std::vector<std::vector<T>> knots_opt(dim);
  bool project = false;
  for (std::size_t d = 0; d < dim; ++d) {
    const auto [a, b] = sf[d].domain();
    knots_full[d] = clamped_product_knots<T>(a, b, meshes[d].breakpoints, bez_h[d], deg_h[d]);
    knots_opt[d] =
        clamped_product_knots<T>(a, b, meshes[d].breakpoints, meshes[d].product, deg_h[d]);
    project = project || meshes[d].product != bez_h[d];
  }
  std::vector<BsplineSpace1D<T>> full_spaces;
  for (std::size_t d = 0; d < dim; ++d) {
    full_spaces.emplace_back(std::span<const T>(knots_full[d]), static_cast<int>(deg_h[d]));
  }
  Bspline<T> h_full(BsplineSpace<T>(std::move(full_spaces)), std::move(ctrl_h), false);
  if (!project) {
    return h_full;
  }
  return map_directions<T>(
      h_full, [](std::size_t /*d*/) { return true; },
      [&](std::size_t d, const BsplineSpace1D<T>& /*space*/,
          mdview<const T, 2> flat) -> std::pair<BsplineSpace1D<T>, NdArray<T>> {
        using Mat = linalg::RowMatrix<T>;
        const int r = static_cast<int>(deg_h[d]);
        const NdArray<T> oslo = oslo_matrix<T>(r, knots_opt[d], knots_full[d]);
        const Mat rhs =
            Eigen::Map<const Mat>(flat.data_handle(), static_cast<Eigen::Index>(flat.extent(0)),
                                  static_cast<Eigen::Index>(flat.extent(1)));
        NdArray<T> ctrl = linalg::lstsq_min_norm<T>(Mat(linalg::as_matrix(oslo)), rhs);
        return {BsplineSpace1D<T>(std::span<const T>(knots_opt[d]), r), std::move(ctrl)};
      });
}

/// Split a B-spline's control points into its first `cols` coordinates and the rest, each a
/// non-rational B-spline over the same space.
template <StorageFloat T>
[[nodiscard]] Bspline<T> select_columns(const Bspline<T>& b, std::size_t first, std::size_t count) {
  const std::size_t cp = b.cp_size();
  const std::span<const T> src = b.control_points();
  const std::size_t rows = src.size() / cp;
  std::vector<std::size_t> shape(b.shape().begin(), b.shape().end() - 1);
  shape.push_back(count);
  NdArray<T> out(shape, for_overwrite);
  for (std::size_t i = 0; i < rows; ++i) {
    std::copy_n(src.data() + i * cp + first, count, out.data() + i * count);
  }
  return Bspline<T>(b.space(), std::move(out), false);
}

/// Join two B-splines over one space column-wise, `[a | b]`.
template <StorageFloat T>
[[nodiscard]] Bspline<T> join_columns(const Bspline<T>& a, const Bspline<T>& b, bool rational) {
  const std::size_t ca = a.cp_size();
  const std::size_t cb = b.cp_size();
  const std::size_t rows = a.control_points().size() / ca;
  std::vector<std::size_t> shape(a.shape().begin(), a.shape().end() - 1);
  shape.push_back(ca + cb);
  NdArray<T> out(shape, for_overwrite);
  for (std::size_t i = 0; i < rows; ++i) {
    std::copy_n(a.control_points().data() + i * ca, ca, out.data() + i * (ca + cb));
    std::copy_n(b.control_points().data() + i * cb, cb, out.data() + i * (ca + cb) + ca);
  }
  return Bspline<T>(a.space(), std::move(out), rational);
}

/// The homogeneous form of a B-spline: itself when rational, else with a column of unit weights.
template <StorageFloat T>
[[nodiscard]] Bspline<T> homogeneous(const Bspline<T>& b) {
  if (b.rational()) {
    return b;
  }
  const std::size_t cp = b.cp_size();
  const std::size_t rows = b.control_points().size() / cp;
  std::vector<std::size_t> shape(b.shape().begin(), b.shape().end() - 1);
  shape.push_back(cp + 1);
  NdArray<T> out(shape, for_overwrite);
  for (std::size_t i = 0; i < rows; ++i) {
    std::copy_n(b.control_points().data() + i * cp, cp, out.data() + i * (cp + 1));
    out.data()[i * (cp + 1) + cp] = T(1);
  }
  return Bspline<T>(b.space(), std::move(out), true);
}

}  // namespace detail

/// Compute the exact pointwise product `h = f g` of two B-splines.
///
/// The product has degree `p_d + q_d` per direction and the minimal interior multiplicities (see
/// the file note); a `C^-1` operand gives a `C^-1` product there. Rational operands are multiplied
/// in homogeneous form, a non-rational operand promoted with unit weights when the other is
/// rational. Per direction, either operand open gives an open result, both periodic a periodic
/// one, and otherwise an unclamped one.
///
/// Args:
///     f (const Bspline<T>&): The first operand.
///     g (const Bspline<T>&): The second operand.
///
/// Returns:
///     Bspline<T>: The product.
///
/// Raises:
///     value_error: The operands differ in parametric dimension or in rank, or their parametric
///         domains differ in some direction by more than the larger tolerance of their open forms.
///         The messages are the previous release's (its one-direction and multi-direction paths
///         worded them differently).
template <StorageFloat T>
[[nodiscard]] Bspline<T> multiply(const Bspline<T>& f, const Bspline<T>& g) {
  using detail::ProductBoundary;
  const std::size_t dim = f.dim();
  const bool one = dim == 1;
  if (one && g.dim() != 1) {
    throw value_error("g must be a 1D B-spline, got dim=" + std::to_string(g.dim()));
  }
  if (!one && g.dim() != dim) {
    throw value_error("f and g must have the same parametric dimension. Got f.dim=" +
                      std::to_string(dim) + " and g.dim=" + std::to_string(g.dim()) + ".");
  }
  if (f.rank() != g.rank()) {
    throw value_error(one ? "f and g must have the same rank, got " + std::to_string(f.rank()) +
                                " and " + std::to_string(g.rank())
                          : "f and g must have the same rank. Got " + std::to_string(f.rank()) +
                                " and " + std::to_string(g.rank()) + ".");
  }
  const BsplineSpace<T> space_f = f.space();
  const BsplineSpace<T> space_g = g.space();
  const std::vector<BsplineSpace1D<T>>& sf = space_f.spaces();
  const std::vector<BsplineSpace1D<T>>& sg = space_g.spaces();
  std::vector<ProductBoundary> target(dim);
  bool f_open = true;
  bool g_open = true;
  for (std::size_t d = 0; d < dim; ++d) {
    const bool fo = sf[d].has_open_knots() && !sf[d].periodic();
    const bool go = sg[d].has_open_knots() && !sg[d].periodic();
    target[d] = (fo || go)                               ? ProductBoundary::open
                : (sf[d].periodic() && sg[d].periodic()) ? ProductBoundary::periodic
                                                         : ProductBoundary::nonopen;
    f_open = f_open && fo;
    g_open = g_open && go;
  }
  const Bspline<T> fo = f_open ? f : to_open_bspline<T>(f);
  const Bspline<T> go = g_open ? g : to_open_bspline<T>(g);

  const BsplineSpace<T> open_f = fo.space();
  const BsplineSpace<T> open_g = go.space();
  const std::vector<BsplineSpace1D<T>>& of = open_f.spaces();
  const std::vector<BsplineSpace1D<T>>& og = open_g.spaces();
  double tol = 0.0;
  for (std::size_t d = 0; d < dim; ++d) {
    tol = (std::max)({tol, of[d].tolerance(), og[d].tolerance()});
  }
  for (std::size_t d = 0; d < dim; ++d) {
    const auto [fa, fb] = of[d].domain();
    const auto [ga, gb] = og[d].domain();
    const bool differ = std::abs(static_cast<double>(fa) - static_cast<double>(ga)) > tol ||
                        std::abs(static_cast<double>(fb) - static_cast<double>(gb)) > tol;
    if (differ) {
      const std::string where = one ? "" : " in direction " + std::to_string(d);
      throw value_error("f and g must share the same parametric domain" + where + ". Got (" +
                        py_repr(static_cast<double>(fa)) + ", " + py_repr(static_cast<double>(fb)) +
                        ") and (" + py_repr(static_cast<double>(ga)) + ", " +
                        py_repr(static_cast<double>(gb)) + ").");
    }
  }

  Bspline<T> h_open = [&] {
    if (!fo.rational() && !go.rational()) {
      return detail::multiply_open_nonrational<T>(fo, go);
    }
    const Bspline<T> fh = detail::homogeneous<T>(fo);
    const Bspline<T> gh = detail::homogeneous<T>(go);
    const std::size_t n = fh.cp_size() - 1;
    const Bspline<T> num = detail::multiply_open_nonrational<T>(
        detail::select_columns<T>(fh, 0, n), detail::select_columns<T>(gh, 0, n));
    const Bspline<T> den = detail::multiply_open_nonrational<T>(
        detail::select_columns<T>(fh, n, 1), detail::select_columns<T>(gh, n, 1));
    return detail::join_columns<T>(num, den, true);
  }();

  const bool any = std::any_of(target.begin(), target.end(),
                               [](ProductBoundary t) { return t != ProductBoundary::open; });
  if (!any) {
    return h_open;
  }
  return detail::map_directions<T>(
      h_open, [&](std::size_t d) { return target[d] != ProductBoundary::open; },
      [&](std::size_t d, const BsplineSpace1D<T>& space,
          mdview<const T, 2> flat) -> std::pair<BsplineSpace1D<T>, NdArray<T>> {
        return detail::convert_product_boundary<T>(space, flat, target[d], sf[d], sg[d], one);
      });
}

}  // namespace pantr::bspline
