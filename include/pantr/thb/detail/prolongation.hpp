/// @file prolongation.hpp
/// @brief The prolongation matrix between a THB space and a refinement of it, in CSR form: one
/// local least-squares solve per coarse function.
///
/// Each coarse function is matched against only the fine functions whose support covers its own,
/// all expressed in the deepest tensor-product level present there (not the global finest one), and
/// the coefficients are the solution of a small least-squares problem (Garau and Vazquez 2018;
/// D'Angella et al. 2018). Functions away from the refinement give trivial columns, so cost and
/// sparsity follow the refined region.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed) unless a function says so. For
///     general use, call `pantr::thb::THBSplineSpace::prolongation_csr`.
#pragma once

#include "pantr/bspline/detail/support_1d.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/format.hpp"
#include "pantr/core/scalar.hpp"
#include "pantr/grid/detail/py_text.hpp"
#include "pantr/grid/hierarchical_grid.hpp"
#include "pantr/linalg/dense.hpp"
#include "pantr/parallel/parallel_for.hpp"
#include "pantr/parallel/work_model.hpp"
#include "pantr/thb/detail/rep.hpp"
#include "pantr/thb/detail/truncation.hpp"
#include "pantr/thb/tolerances.hpp"
#include "pantr/thb/types.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace pantr::thb::detail {

/// Spell a regularity tuple as Python spells `repr(tuple)` of ints and `None`: `(None, 1)`, `(2,)`.
///
/// Args:
///     reg (std::span<const std::optional<int>>): The entries.
///
/// Returns:
///     std::string: The spelling.
[[nodiscard]] inline std::string py_repr_regularity(std::span<const std::optional<int>> reg) {
  std::string out = "(";
  for (std::size_t i = 0; i < reg.size(); ++i) {
    out += (i > 0 ? ", " : "") + (reg[i].has_value() ? std::to_string(*reg[i]) : "None");
  }
  if (reg.size() == 1) {
    out += ',';
  }
  return out + ")";
}

/// Check that `fine` is structurally a refinement of `coarse`.
///
/// Everything decidable without assembling: the dimension, the truncation mode, the subdivision
/// factor, the regularity, the level count and the root knot vectors (compared exactly). That the
/// coarse basis is reproducible in the fine one is the residual check of the assembly.
///
/// Args:
///     coarse (const THBSpaceRep<T>&): The coarse space.
///     fine (const THBSpaceRep<T>&): The candidate refinement.
///
/// Raises:
///     value_error: Some property disagrees; the message lists every mismatch found.
template <StorageFloat T>
void check_is_refinement(const THBSpaceRep<T>& coarse, const THBSpaceRep<T>& fine) {
  const std::size_t dim = coarse.root_space.dim();
  const auto spell = [](bool b) { return std::string(b ? "True" : "False"); };
  std::vector<std::string> mismatches;
  if (fine.root_space.dim() != dim) {
    mismatches.push_back("dim: self=" + std::to_string(dim) +
                         " vs fine=" + std::to_string(fine.root_space.dim()));
  }
  if (fine.truncate != coarse.truncate) {
    mismatches.push_back("truncate: self=" + spell(coarse.truncate) +
                         " vs fine=" + spell(fine.truncate));
  }
  const std::span<const std::int64_t> cf = coarse.hier.factor();
  const std::span<const std::int64_t> ff = fine.hier.factor();
  if (!std::equal(cf.begin(), cf.end(), ff.begin(), ff.end())) {
    mismatches.push_back("factor: self=" + grid::detail::py_repr_ints(cf) +
                         " vs fine=" + grid::detail::py_repr_ints(ff));
  }
  if (fine.regularity != coarse.regularity) {
    mismatches.push_back("regularity: self=" + py_repr_regularity(coarse.regularity) +
                         " vs fine=" + py_repr_regularity(fine.regularity));
  }
  if (fine.level_spaces.size() < coarse.level_spaces.size()) {
    mismatches.push_back("fine.num_levels=" + std::to_string(fine.level_spaces.size()) +
                         " < self.num_levels=" + std::to_string(coarse.level_spaces.size()));
  }
  if (fine.root_space.dim() == dim) {
    bool same = true;
    for (std::size_t k = 0; k < dim && same; ++k) {
      const std::span<const T> a = coarse.root_space.spaces()[k].knots();
      const std::span<const T> b = fine.root_space.spaces()[k].knots();
      same = std::equal(a.begin(), a.end(), b.begin(), b.end());
    }
    if (!same) {
      mismatches.emplace_back("root knot vectors differ");
    }
  }
  if (!mismatches.empty()) {
    std::string message = "fine must be a refinement of this space; mismatches: ";
    for (std::size_t i = 0; i < mismatches.size(); ++i) {
      message += (i > 0 ? "; " : "") + mismatches[i];
    }
    throw value_error(message + ".");
  }
}

/// The root-cell support box of every active function of a space, and the level each is expressed
/// in.
///
/// Attributes:
///     dim (std::size_t): Number of directions.
///     rep_level (std::vector<std::int64_t>): Per dof, the representation level (the stored level
///         of a truncated function, else its own level).
///     lo (std::vector<std::int64_t>): Row-major `(dofs, dim)` first root cell of the support.
///     hi (std::vector<std::int64_t>): Row-major `(dofs, dim)` last root cell of the support
///         (inclusive).
struct RootBoxes {
  std::size_t dim = 0;
  std::vector<std::int64_t> rep_level;
  std::vector<std::int64_t> lo;
  std::vector<std::int64_t> hi;
};

/// Compute the root-cell support box of every active function of a space.
///
/// A function's support is a box of cells of its own level; the root cells it meets are those whose
/// level-`l` subdivision it covers, found by integer division with the level's factor power.
///
/// Args:
///     space (const THBSpaceRep<T>&): The space.
///
/// Returns:
///     RootBoxes: One entry per global dof.
template <StorageFloat T>
[[nodiscard]] RootBoxes func_root_boxes(const THBSpaceRep<T>& space) {
  const std::size_t dim = space.root_space.dim();
  const std::span<const std::int64_t> factor = space.hier.factor();
  const auto n = static_cast<std::size_t>(space.func_offset.back());
  RootBoxes out;
  out.dim = dim;
  out.rep_level.assign(n, 0);
  out.lo.assign(n * dim, 0);
  out.hi.assign(n * dim, 0);
  std::vector<std::int64_t> scale(dim);
  std::vector<std::int64_t> multi(dim);
  for (std::size_t level = 0; level < space.active.size(); ++level) {
    for (std::size_t k = 0; k < dim; ++k) {
      scale[k] = grid::detail::checked_pow(factor[k], static_cast<std::int64_t>(level));
    }
    const std::vector<std::int64_t>& nb = space.num_basis[level];
    for (std::size_t pos = 0; pos < space.active[level].size(); ++pos) {
      const auto dof = static_cast<std::size_t>(space.func_offset[level]) + pos;
      std::int64_t rest = space.active[level][pos];
      for (std::size_t k = dim; k-- > 0;) {
        multi[k] = rest % nb[k];
        rest /= nb[k];
      }
      for (std::size_t k = 0; k < dim; ++k) {
        const bspline::detail::Support1D& s = space.support[level][k];
        out.lo[dof * dim + k] = s.first_cell[static_cast<std::size_t>(multi[k])] / scale[k];
        out.hi[dof * dim + k] = s.last_cell[static_cast<std::size_t>(multi[k])] / scale[k];
      }
      const std::int64_t at = space.trunc_index[dof];
      out.rep_level[dof] = at < 0 ? static_cast<std::int64_t>(level)
                                  : space.trunc[static_cast<std::size_t>(at)].rep_level;
    }
  }
  return out;
}

/// Call `visit(flat)` for the flat (row-major) id of every root cell of the inclusive box
/// `[lo, hi]`, in increasing order.
///
/// Args:
///     lo (std::span<const std::int64_t>): Per-direction first cell.
///     hi (std::span<const std::int64_t>): Per-direction last cell, at least `lo`.
///     root_cells (std::span<const std::int64_t>): Per-direction root cell counts.
///     visit (F&&): Called once per cell.
template <class F>
void for_each_root_cell(std::span<const std::int64_t> lo, std::span<const std::int64_t> hi,
                        std::span<const std::int64_t> root_cells, F&& visit) {
  const std::size_t dim = lo.size();
  std::vector<std::int64_t> idx(lo.begin(), lo.end());
  bool done = false;
  while (!done) {
    std::int64_t flat = 0;
    for (std::size_t k = 0; k < dim; ++k) {
      flat = flat * root_cells[k] + idx[k];
    }
    visit(flat);
    done = true;
    for (std::size_t k = dim; k-- > 0;) {
      if (++idx[k] <= hi[k]) {
        done = false;
        break;
      }
      idx[k] = lo[k];
    }
  }
}

/// Get the level that owns global dof `dof`.
///
/// Args:
///     space (const THBSpaceRep<T>&): The space.
///     dof (std::int64_t): A dof in `[0, num_total_basis)`.
///
/// Returns:
///     std::size_t: The level whose dof range contains `dof`.
template <StorageFloat T>
[[nodiscard]] std::size_t dof_level(const THBSpaceRep<T>& space, std::int64_t dof) {
  const auto it = std::upper_bound(space.func_offset.begin(), space.func_offset.end(), dof);
  return static_cast<std::size_t>(it - space.func_offset.begin()) - 1;
}

/// A function expressed in a tensor-product basis: flat indices and coefficients.
///
/// Attributes:
///     flats (std::vector<std::int64_t>): Flat (row-major) indices in the level's basis.
///     coeffs (std::vector<double>): The coefficient of each.
struct TpColumn {
  std::vector<std::int64_t> flats;
  std::vector<double> coeffs;
};

/// Express active function `dof` of `space` in the level-`target` tensor-product basis.
///
/// Takes the function's own representation (a single B-spline when untruncated, the stored box
/// otherwise) and refines it by the two-scale matrices of `fine`, with no truncation, up to
/// `target`.
///
/// Args:
///     space (const THBSpaceRep<T>&): The space owning `dof`.
///     dof (std::int64_t): A global dof of `space`.
///     fine (const THBSpaceRep<T>&): The space whose two-scale matrices and level spaces are used.
///     target (std::int64_t): A level of `fine`, at least the function's representation level.
///
/// Returns:
///     TpColumn: The function's coefficients in `fine`'s level-`target` basis.
///
/// Raises:
///     value_error: A refinement step meets a zero two-scale slice (see `refine_box`).
template <StorageFloat T>
[[nodiscard]] TpColumn tp_column(const THBSpaceRep<T>& space, std::int64_t dof,
                                 const THBSpaceRep<T>& fine, std::int64_t target) {
  const std::size_t dim = space.root_space.dim();
  const std::size_t level = dof_level(space, dof);
  const auto pos = static_cast<std::size_t>(dof - space.func_offset[level]);
  const std::int64_t at = space.trunc_index[static_cast<std::size_t>(dof)];
  CoeffBox box;
  std::int64_t start = static_cast<std::int64_t>(level);
  if (at < 0) {
    box.lo.resize(dim);
    box.shape.assign(dim, 1);
    std::int64_t rest = space.active[level][pos];
    for (std::size_t k = dim; k-- > 0;) {
      box.lo[k] = rest % space.num_basis[level][k];
      rest /= space.num_basis[level][k];
    }
    box.coeffs.assign(1, 1.0);
  } else {
    const TruncatedFunction& f = space.trunc[static_cast<std::size_t>(at)];
    start = f.rep_level;
    box.lo = f.box_lo;
    box.shape = f.box_shape;
    box.coeffs = f.coeffs;
  }
  for (std::int64_t l = start; l < target; ++l) {
    box = refine_box(box, fine.two_scale[static_cast<std::size_t>(l)]);
  }
  const std::vector<std::int64_t>& nb = fine.num_basis[static_cast<std::size_t>(target)];
  TpColumn out;
  out.coeffs = std::move(box.coeffs);
  out.flats.resize(out.coeffs.size());
  std::vector<std::size_t> idx(dim, 0);
  for (std::size_t i = 0; i < out.flats.size(); ++i) {
    std::int64_t flat = 0;
    for (std::size_t k = 0; k < dim; ++k) {
      flat = flat * nb[k] + box.lo[k] + static_cast<std::int64_t>(idx[k]);
    }
    out.flats[i] = flat;
    for (std::size_t k = dim; k-- > 0;) {
      if (++idx[k] < box.shape[k]) {
        break;
      }
      idx[k] = 0;
    }
  }
  return out;
}

/// The larger of `acc` and `v`, NaN when either is.
///
/// `std::max(acc, NaN)` returns `acc`, which would let a NaN residual through the gate as a small
/// one.
///
/// Args:
///     acc (double): The running maximum.
///     v (double): The next value.
///
/// Returns:
///     double: `max(acc, v)`, or NaN where `acc` or `v` is NaN.
[[nodiscard]] inline double nan_propagating_max(double acc, double v) {
  return std::isnan(v) ? v : (std::max)(acc, v);
}

/// One column of the prolongation, as solved.
///
/// Attributes:
///     rows (std::vector<std::int64_t>): The fine dofs the column occupies, increasing; empty when
///         no fine function covers the coarse function's support.
///     values (std::vector<double>): The solution on `rows`.
///     residual (double): `max |A x - b|` of the local solve (`max |b|` when `rows` is empty).
///     coarse_val (double): `max |b|`, the largest coefficient of the coarse function.
struct ProlongationColumn {
  std::vector<std::int64_t> rows;
  std::vector<double> values;
  double residual = 0.0;
  double coarse_val = 0.0;
};

/// Everything the column solves read, built once: the support boxes of both spaces and the
/// inverted index from root cells to the fine functions covering them.
///
/// Attributes:
///     coarse_boxes (RootBoxes): Support boxes of the coarse functions.
///     fine_boxes (RootBoxes): Support boxes of the fine functions.
///     root_cells (std::vector<std::int64_t>): Per-direction root cell counts.
///     cell_ptr (std::vector<std::int64_t>): `root cells + 1` offsets into `cell_fine`.
///     cell_fine (std::vector<std::int64_t>): For each root cell, the fine dofs whose support box
///         contains it, increasing.
struct ProlongationIndex {
  RootBoxes coarse_boxes;
  RootBoxes fine_boxes;
  std::vector<std::int64_t> root_cells;
  std::vector<std::int64_t> cell_ptr;
  std::vector<std::int64_t> cell_fine;
};

/// Build the index the column solves read.
///
/// Args:
///     coarse (const THBSpaceRep<T>&): The coarse space.
///     fine (const THBSpaceRep<T>&): A validated refinement of it.
///
/// Returns:
///     ProlongationIndex: The boxes and the cell-to-function index.
template <StorageFloat T>
[[nodiscard]] ProlongationIndex build_prolongation_index(const THBSpaceRep<T>& coarse,
                                                         const THBSpaceRep<T>& fine) {
  const std::size_t dim = coarse.root_space.dim();
  ProlongationIndex index;
  index.coarse_boxes = func_root_boxes(coarse);
  index.fine_boxes = func_root_boxes(fine);
  index.root_cells = coarse.hier.level_cells_per_axis(0);
  std::size_t n_root = 1;
  for (const std::int64_t c : index.root_cells) {
    n_root *= static_cast<std::size_t>(c);
  }
  const std::span<const std::int64_t> root_cells(index.root_cells);
  const std::size_t n_fine = index.fine_boxes.rep_level.size();
  const auto box_of = [dim](const RootBoxes& boxes, std::size_t j) {
    return std::pair{std::span<const std::int64_t>(boxes.lo).subspan(j * dim, dim),
                     std::span<const std::int64_t>(boxes.hi).subspan(j * dim, dim)};
  };
  index.cell_ptr.assign(n_root + 1, 0);
  for (std::size_t j = 0; j < n_fine; ++j) {
    const auto [lo, hi] = box_of(index.fine_boxes, j);
    for_each_root_cell(lo, hi, root_cells, [&](std::int64_t cell) {
      ++index.cell_ptr[static_cast<std::size_t>(cell) + 1];
    });
  }
  for (std::size_t c = 0; c < n_root; ++c) {
    index.cell_ptr[c + 1] += index.cell_ptr[c];
  }
  index.cell_fine.assign(static_cast<std::size_t>(index.cell_ptr[n_root]), 0);
  std::vector<std::int64_t> next(index.cell_ptr.begin(), index.cell_ptr.end() - 1);
  for (std::size_t j = 0; j < n_fine; ++j) {
    const auto [lo, hi] = box_of(index.fine_boxes, j);
    for_each_root_cell(lo, hi, root_cells, [&](std::int64_t cell) {
      index.cell_fine[static_cast<std::size_t>(next[static_cast<std::size_t>(cell)]++)] =
          static_cast<std::int64_t>(j);
    });
  }
  return index;
}

/// Solve the local system that reproduces coarse function `i` in the fine basis.
///
/// The candidates are the fine functions whose support box shares a root cell with the coarse
/// function's. All are expressed in the deepest representation level among them and the coarse
/// function's own; the unknowns are the candidates' coefficients, the right-hand side the coarse
/// function's coefficients, and the system is solved in the least-squares sense. The columns of the
/// system are linearly independent hierarchical functions of a valid space, so it has full column
/// rank and is solved by Householder QR with column pivoting (`linalg::try_lstsq_full_rank_qr`,
/// which answers only where a rank certificate holds). Where it declines, a rank-deficient system
/// that only a tiling `HierarchicalGrid::from_blocks` accepted without validating can reach, the
/// minimum-norm solution by `linalg::lstsq_min_norm_jacobi` is used. Both are backward stable,
/// which the residual gate that follows assumes (see `prolongation_residual_tolerance`, whose
/// factor is admitted rather than derived); the divide-and-conquer SVD of `lstsq_min_norm` is not
/// on every local system.
///
/// Args:
///     coarse (const THBSpaceRep<T>&): The coarse space.
///     fine (const THBSpaceRep<T>&): The refinement.
///     index (const ProlongationIndex&): The boxes and the cell index.
///     i (std::int64_t): The coarse dof.
///
/// Returns:
///     ProlongationColumn: The solved column and its diagnostics.
///
/// Raises:
///     value_error: A refinement step meets a zero two-scale slice, or the system has a NaN or an
///         infinite entry (see `linalg::lstsq_min_norm_jacobi`).
template <StorageFloat T>
[[nodiscard]] ProlongationColumn solve_prolongation_column(const THBSpaceRep<T>& coarse,
                                                           const THBSpaceRep<T>& fine,
                                                           const ProlongationIndex& index,
                                                           std::int64_t i) {
  const std::size_t dim = coarse.root_space.dim();
  const auto iu = static_cast<std::size_t>(i);
  const std::span<const std::int64_t> root_cells(index.root_cells);
  ProlongationColumn col;
  for_each_root_cell(std::span<const std::int64_t>(index.coarse_boxes.lo).subspan(iu * dim, dim),
                     std::span<const std::int64_t>(index.coarse_boxes.hi).subspan(iu * dim, dim),
                     root_cells, [&](std::int64_t cell) {
                       const auto c = static_cast<std::size_t>(cell);
                       col.rows.insert(col.rows.end(), index.cell_fine.begin() + index.cell_ptr[c],
                                       index.cell_fine.begin() + index.cell_ptr[c + 1]);
                     });
  std::sort(col.rows.begin(), col.rows.end());
  col.rows.erase(std::unique(col.rows.begin(), col.rows.end()), col.rows.end());
  const std::vector<std::int64_t>& candidates = col.rows;

  std::int64_t level = index.coarse_boxes.rep_level[iu];
  for (const std::int64_t j : candidates) {
    level = (std::max)(level, index.fine_boxes.rep_level[static_cast<std::size_t>(j)]);
  }
  const TpColumn target = tp_column(coarse, i, fine, level);
  for (const double v : target.coeffs) {
    col.coarse_val = (std::max)(col.coarse_val, std::abs(v));
  }
  std::vector<TpColumn> cols;
  cols.reserve(candidates.size());
  for (const std::int64_t j : candidates) {
    cols.push_back(tp_column(fine, j, fine, level));
  }
  std::vector<std::int64_t> row_flats = target.flats;
  for (const TpColumn& c : cols) {
    row_flats.insert(row_flats.end(), c.flats.begin(), c.flats.end());
  }
  std::sort(row_flats.begin(), row_flats.end());
  row_flats.erase(std::unique(row_flats.begin(), row_flats.end()), row_flats.end());
  const auto row_of = [&](std::int64_t flat) {
    return static_cast<Eigen::Index>(std::lower_bound(row_flats.begin(), row_flats.end(), flat) -
                                     row_flats.begin());
  };
  const auto n_rows = static_cast<Eigen::Index>(row_flats.size());
  linalg::RowMatrix<double> rhs = linalg::RowMatrix<double>::Zero(n_rows, 1);
  for (std::size_t t = 0; t < target.flats.size(); ++t) {
    rhs(row_of(target.flats[t]), 0) = target.coeffs[t];
  }
  // No fine function covers the coarse support: not possible on a valid tiling, but
  // `HierarchicalGrid::from_blocks` takes its blocks as a valid tiling without checking it. The
  // column is then empty and its whole right-hand side is residual.
  if (candidates.empty()) {
    col.residual = rhs.cwiseAbs().template maxCoeff<Eigen::PropagateNaN>();
    return col;
  }
  const auto n_cols = static_cast<Eigen::Index>(candidates.size());
  linalg::RowMatrix<double> amat = linalg::RowMatrix<double>::Zero(n_rows, n_cols);
  for (Eigen::Index cj = 0; cj < n_cols; ++cj) {
    const TpColumn& c = cols[static_cast<std::size_t>(cj)];
    for (std::size_t t = 0; t < c.flats.size(); ++t) {
      amat(row_of(c.flats[t]), cj) = c.coeffs[t];
    }
  }
  std::optional<NdArray<double>> full_rank = linalg::try_lstsq_full_rank_qr<double>(amat, rhs);
  const NdArray<double> sol =
      full_rank ? std::move(*full_rank) : linalg::lstsq_min_norm_jacobi<double>(amat, rhs);
  col.values.assign(sol.data(), sol.data() + n_cols);
  for (Eigen::Index r = 0; r < n_rows; ++r) {
    double acc = 0.0;
    for (Eigen::Index cj = 0; cj < n_cols; ++cj) {
      acc += amat(r, cj) * col.values[static_cast<std::size_t>(cj)];
    }
    col.residual = nan_propagating_max(col.residual, std::abs(acc - rhs(r, 0)));
  }
  return col;
}

/// Get the work of one column solve, which sizes the chunks of the parallel loop over columns.
///
/// A column with `K` candidates builds `K` tensor-product columns of at least `V = prod(degree +
/// 1)` coefficients (a function refined only grows its box), each refined once per direction, and
/// solves a system of `R >= K V` rows and `K` columns by Householder QR, whose leading cost is `2
/// R K^2 - 2 K^3 / 3` operations (Golub and Van Loan, *Matrix Computations*, 4th ed., section
/// 5.2.1), taken as `2 R K^2`; the rank certificate's triangular inverse, `K^3 / 3`, and the column
/// pivoting's norm updates, `O(R K)`, are below it for `R >= K`. The rare rank-deficient column
/// that falls back to the SVD costs more and is not priced. `K` is the mean over the coarse columns
/// of the candidate count with a function counted once per root cell it shares with the coarse
/// support, an upper bound on the distinct count. The lower bound on `R` and the upper bound on `K`
/// pull in opposite directions; the estimate is an order of magnitude, an admitted heuristic, and
/// it only chooses how many columns one chunk takes. It is floored at one operation, the least
/// `parallel::WorkEstimate` accepts.
///
/// Args:
///     mean_candidates (double): The mean candidate bound per coarse column.
///     dim (std::size_t): Number of directions.
///     degrees (std::span<const int>): Per-direction degrees.
///
/// Returns:
///     parallel::WorkEstimate: Operations per column.
[[nodiscard]] inline parallel::WorkEstimate prolongation_column_work(double mean_candidates,
                                                                     std::size_t dim,
                                                                     std::span<const int> degrees) {
  double v = 1.0;
  for (const int p : degrees) {
    v *= static_cast<double>(p) + 1.0;
  }
  const double k = mean_candidates;
  return parallel::WorkEstimate{
      (std::max)(1.0, k * v * static_cast<double>(dim) + 2.0 * (k * v) * k * k)};
}

/// Assemble the prolongation from a space to a validated refinement of it.
///
/// The columns are independent, so they are solved in parallel over the coarse functions, each into
/// its own buffer; the matrix is then gathered serially in column order, which makes it, and so
/// every stored value, the same for every thread count. The residual and the coarse magnitude are
/// maxima over the columns, which do not depend on the order.
///
/// Args:
///     coarse (const THBSpaceRep<T>&): The coarse space.
///     fine (const THBSpaceRep<T>&): A refinement of it (see `check_is_refinement`).
///
/// Returns:
///     ProlongationCsr: The matrix of shape `(fine dofs, coarse dofs)`.
///
/// Raises:
///     value_error: A column cannot reproduce its coarse function (residual above
///         `prolongation_residual_tolerance`: `fine` is not a refinement), a refinement step meets
///         a zero two-scale slice, or the solver refuses.
template <StorageFloat T>
[[nodiscard]] ProlongationCsr assemble_prolongation(const THBSpaceRep<T>& coarse,
                                                    const THBSpaceRep<T>& fine) {
  const std::size_t dim = coarse.root_space.dim();
  const ProlongationIndex index = build_prolongation_index(coarse, fine);
  const std::int64_t n_coarse = coarse.func_offset.back();
  const std::int64_t n_fine = fine.func_offset.back();
  const auto n_cols = static_cast<std::size_t>(n_coarse);

  double candidate_bound = 0.0;
  for (std::size_t i = 0; i < n_cols; ++i) {
    for_each_root_cell(std::span<const std::int64_t>(index.coarse_boxes.lo).subspan(i * dim, dim),
                       std::span<const std::int64_t>(index.coarse_boxes.hi).subspan(i * dim, dim),
                       std::span<const std::int64_t>(index.root_cells), [&](std::int64_t cell) {
                         const auto c = static_cast<std::size_t>(cell);
                         candidate_bound +=
                             static_cast<double>(index.cell_ptr[c + 1] - index.cell_ptr[c]);
                       });
  }
  const double mean_candidates = candidate_bound / static_cast<double>(n_cols);
  const parallel::WorkEstimate work =
      prolongation_column_work(mean_candidates, dim, coarse.degrees);

  std::vector<ProlongationColumn> columns(n_cols);
  parallel::parallel_for(n_cols, work, [&](std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; ++i) {
      columns[i] = solve_prolongation_column(coarse, fine, index, static_cast<std::int64_t>(i));
    }
  });

  double max_residual = 0.0;
  double max_coarse_val = 0.0;
  for (const ProlongationColumn& c : columns) {
    max_residual = nan_propagating_max(max_residual, c.residual);
    max_coarse_val = (std::max)(max_coarse_val, c.coarse_val);
  }
  const double residual_tol = prolongation_residual_tolerance(max_coarse_val);
  // Written so that a NaN residual fails the gate: a NaN column must not pass as a refinement.
  if (!(max_residual <= residual_tol)) {
    throw value_error("fine is not a refinement of this space (prolongation residual " +
                      py_format_exp(max_residual, 2) + ", above " + py_format_exp(residual_tol, 2) +
                      ").");
  }

  ProlongationCsr out;
  out.rows = n_fine;
  out.cols = n_coarse;
  out.indptr.assign(static_cast<std::size_t>(n_fine) + 1, 0);
  for (const ProlongationColumn& c : columns) {
    for (const std::int64_t r : c.rows) {
      ++out.indptr[static_cast<std::size_t>(r) + 1];
    }
  }
  for (std::size_t r = 0; r < static_cast<std::size_t>(n_fine); ++r) {
    out.indptr[r + 1] += out.indptr[r];
  }
  out.indices.assign(static_cast<std::size_t>(out.indptr.back()), 0);
  out.data.assign(static_cast<std::size_t>(out.indptr.back()), 0.0);
  std::vector<std::int64_t> next(out.indptr.begin(), out.indptr.end() - 1);
  for (std::size_t i = 0; i < n_cols; ++i) {
    const ProlongationColumn& c = columns[i];
    for (std::size_t t = 0; t < c.rows.size(); ++t) {
      const auto at = static_cast<std::size_t>(next[static_cast<std::size_t>(c.rows[t])]++);
      out.indices[at] = static_cast<std::int64_t>(i);
      out.data[at] = c.values[t];
    }
  }
  return out;
}

}  // namespace pantr::thb::detail
