/// @file partition_graph.hpp
/// @brief `partition_graph`: split a `CouplingGraph` into rank subdomains, and the `Spectral`
/// partitioner (recursive Fiedler bisection).
///
/// A partitioner models `GraphPartitioner`: it takes a graph, a part count and an optional
/// activity mask and returns a `grid::Partition` (inactive cells get owner -1) with a
/// `PartitionInfo` saying how it got there. `Spectral` is built in and needs nothing beyond Eigen
/// and Spectra; METIS is a provider (`pantr/providers/metis/partitioner.hpp`, built with
/// `-DPANTR_WITH_METIS=ON`).
///
/// Determinism: nothing here runs in parallel, so a partition does not depend on the thread count.
/// The Fiedler vector's sign is fixed by making its largest-magnitude entry (the first, on a tie)
/// positive, ties in the ordering keep the vertices' current order (a stable sort), and the
/// iterative solver starts from a fixed vector. A partition is not reproducible across builds,
/// though: the eigensolvers' last bits depend on the SIMD width they are compiled for (each ISA
/// rung of the Python extension is one such build), and wherever the Fiedler vector is
/// ill-determined that decides the partition: a repeated or near-repeated second eigenvalue, or
/// two entries equal up to rounding (a graph symmetric under reversing its vertex order, whose
/// parts can then come out mirrored), or several entries equal up to rounding where the order is
/// split. With three parts or more each part split again is such a graph too, so a graph that is
/// not symmetric itself can still have a symmetric part.
#pragma once

#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/grid/partition.hpp"
#include "pantr/partition/coupling_graph.hpp"

#include <Spectra/SymEigsSolver.h>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pantr::partition {

/// Subgraphs with at most this many vertices take the dense eigensolver.
inline constexpr std::int64_t dense_fiedler_max = 512;

/// How a partitioner reached its result: one count per subgraph ordering.
///
/// Attributes:
///     dense_solves (std::int64_t): Connected subgraphs of at most `dense_max` vertices, ordered
///         by the dense eigensolver.
///     lanczos_solves (std::int64_t): Larger connected subgraphs whose Lanczos solve converged.
///     dense_fallbacks (std::int64_t): Larger connected subgraphs whose Lanczos solve did not
///         converge, ordered by the dense eigensolver instead.
///     component_orders (std::int64_t): Disconnected subgraphs, ordered by component.
struct PartitionInfo {
  std::int64_t dense_solves = 0;
  std::int64_t lanczos_solves = 0;
  std::int64_t dense_fallbacks = 0;
  std::int64_t component_orders = 0;
};

/// A partition and how it was computed.
///
/// Attributes:
///     partition (grid::Partition): The per-cell owners.
///     info (PartitionInfo): The partitioner's account of its work.
struct [[nodiscard]] GraphPartition {
  grid::Partition partition;
  PartitionInfo info;
};

/// A graph partitioner: `p.partition(graph, n_parts, active)` returns a `GraphPartition`.
///
/// `active` is one flag per vertex (nonzero is active), or no value for all active; the
/// partitioner validates its arguments as `partition_graph` documents.
template <class P>
concept GraphPartitioner = requires(const P& p, const CouplingGraph& graph, std::int64_t n_parts,
                                    std::optional<std::span<const std::uint8_t>> active) {
  { p.partition(graph, n_parts, active) } -> std::same_as<GraphPartition>;
};

namespace detail {

/// The active subgraph of a coupling graph, renumbered.
///
/// Attributes:
///     active_idx (std::vector<std::int64_t>): The active vertices, ascending; vertex `a` of the
///         subgraph is `active_idx[a]` of the graph.
///     xadj (std::vector<std::int64_t>): Row pointers of the subgraph.
///     adjncy (std::vector<std::int64_t>): Neighbors in subgraph numbering, ascending per row (a
///         stable sort, so repeated entries keep their order).
///     edge_weights (std::vector<std::int64_t>): Weights aligned with `adjncy`.
///     vertex_weights (std::vector<double>): Weights of the active vertices.
struct ActiveSubgraph {
  std::vector<std::int64_t> active_idx;
  std::vector<std::int64_t> xadj;
  std::vector<std::int64_t> adjncy;
  std::vector<std::int64_t> edge_weights;
  std::vector<double> vertex_weights;
};

/// Check a partition request and extract the active subgraph.
///
/// The checks and their order are `partition_graph`'s; every partitioner calls this first.
///
/// Args:
///     graph (const CouplingGraph&): The graph.
///     n_parts (std::int64_t): Number of parts.
///     active (std::optional<std::span<const std::uint8_t>>): One flag per vertex, or none.
///
/// Returns:
///     ActiveSubgraph: The active vertices and the edges among them.
///
/// Raises:
///     value_error: `n_parts < 1`; `active` has the wrong length or marks no vertex; or `n_parts`
///         exceeds the number of active vertices.
[[nodiscard]] inline ActiveSubgraph select_active(
    const CouplingGraph& graph, std::int64_t n_parts,
    std::optional<std::span<const std::uint8_t>> active) {
  if (n_parts < 1) {
    throw value_error("n_parts must be >= 1; got " + std::to_string(n_parts) + ".");
  }
  const auto n = static_cast<std::size_t>(graph.num_vertices());
  ActiveSubgraph sub;
  std::vector<std::int64_t> local_of(n, -1);
  if (active.has_value()) {
    if (active->size() != n) {
      throw value_error(shape_message("cell_active", n, active->size()));
    }
    for (std::size_t v = 0; v < n; ++v) {
      if ((*active)[v] != 0) {
        local_of[v] = static_cast<std::int64_t>(sub.active_idx.size());
        sub.active_idx.push_back(static_cast<std::int64_t>(v));
      }
    }
    if (sub.active_idx.empty()) {
      throw value_error("cell_active must mark at least one cell active.");
    }
  } else {
    sub.active_idx.resize(n);
    std::iota(sub.active_idx.begin(), sub.active_idx.end(), std::int64_t{0});
    std::iota(local_of.begin(), local_of.end(), std::int64_t{0});
  }
  const std::size_t n_active = sub.active_idx.size();
  if (n_parts > static_cast<std::int64_t>(n_active)) {
    throw value_error("n_parts=" + std::to_string(n_parts) +
                      " exceeds the number of active cells (" + std::to_string(n_active) +
                      "); cannot assign every rank a cell.");
  }

  const auto xadj = graph.xadj();
  const auto adjncy = graph.adjncy();
  const auto ew = graph.edge_weights();
  const auto vw = graph.vertex_weights();
  sub.xadj.assign(n_active + 1, 0);
  sub.vertex_weights.resize(n_active);
  sub.adjncy.reserve(adjncy.size());
  sub.edge_weights.reserve(adjncy.size());
  std::vector<std::pair<std::int64_t, std::int64_t>> row;
  for (std::size_t a = 0; a < n_active; ++a) {
    const auto v = static_cast<std::size_t>(sub.active_idx[a]);
    sub.vertex_weights[a] = vw[v];
    const std::size_t first = sub.adjncy.size();
    for (auto k = static_cast<std::size_t>(xadj[v]); k < static_cast<std::size_t>(xadj[v + 1]);
         ++k) {
      const std::int64_t col = local_of[static_cast<std::size_t>(adjncy[k])];
      if (col >= 0) {
        sub.adjncy.push_back(col);
        sub.edge_weights.push_back(ew[k]);
      }
    }
    // Renumbering keeps the order, so a row given ascending (as `coupling_graph` builds it) stays
    // ascending; only a row given out of order is sorted.
    const auto row_begin = sub.adjncy.begin() + static_cast<std::ptrdiff_t>(first);
    if (!std::is_sorted(row_begin, sub.adjncy.end())) {
      row.clear();
      for (std::size_t k = first; k < sub.adjncy.size(); ++k) {
        row.emplace_back(sub.adjncy[k], sub.edge_weights[k]);
      }
      std::stable_sort(row.begin(), row.end(),
                       [](const auto& x, const auto& y) { return x.first < y.first; });
      for (std::size_t k = first; k < sub.adjncy.size(); ++k) {
        sub.adjncy[k] = row[k - first].first;
        sub.edge_weights[k] = row[k - first].second;
      }
    }
    sub.xadj[a + 1] = static_cast<std::int64_t>(sub.adjncy.size());
  }
  return sub;
}

/// Spread the owners of the active vertices over all vertices, -1 for the inactive ones.
///
/// Args:
///     num_vertices (std::int64_t): Number of vertices of the graph.
///     active_idx (std::span<const std::int64_t>): The active vertices.
///     owner_active (std::span<const std::int32_t>): One owner per active vertex.
///     n_parts (std::int64_t): Number of parts.
///
/// Returns:
///     grid::Partition: The partition.
///
/// Raises:
///     value_error: An owner is outside `[-1, n_parts)`.
///
/// Note:
///     The ids in `active_idx` are assumed to lie in `[0, num_vertices)` and `owner_active` to be
///     as long as `active_idx` (no validation performed); `select_active` returns such ids.
[[nodiscard]] inline grid::Partition scatter_owners(std::int64_t num_vertices,
                                                    std::span<const std::int64_t> active_idx,
                                                    std::span<const std::int32_t> owner_active,
                                                    std::int64_t n_parts) {
  std::vector<std::int32_t> owner(static_cast<std::size_t>(num_vertices), -1);
  for (std::size_t a = 0; a < active_idx.size(); ++a) {
    owner[static_cast<std::size_t>(active_idx[a])] = owner_active[a];
  }
  return grid::Partition(std::move(owner), n_parts);
}

/// A connected subgraph's Laplacian `L = D - W`, in CSR form over local vertex ids.
///
/// `W` holds the off-diagonal entries as given (repeated entries summed by the products, a self
/// loop dropped); `D` is the column sums of `W`, as `scipy.sparse.csgraph.laplacian` forms it, so
/// a symmetric graph gets its weighted degrees.
///
/// Attributes:
///     ptr (std::vector<std::size_t>): Row pointers.
///     col (std::vector<std::size_t>): Columns of the off-diagonal entries.
///     val (std::vector<double>): Off-diagonal values (`-w`).
///     diag (std::vector<double>): The diagonal `D`.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
///     For general use, call `Spectral::partition` instead.
struct Laplacian {
  std::vector<std::size_t> ptr;
  std::vector<std::size_t> col;
  std::vector<double> val;
  std::vector<double> diag;

  /// Compute `y = L x`.
  ///
  /// Args:
  ///     x (const double*): `diag.size()` entries.
  ///     y (double*): `diag.size()` entries, overwritten.
  void apply(const double* x, double* y) const noexcept {
    const std::size_t m = diag.size();
    for (std::size_t i = 0; i < m; ++i) {
      double acc = diag[i] * x[i];
      for (std::size_t k = ptr[i]; k < ptr[i + 1]; ++k) {
        acc += val[k] * x[col[k]];
      }
      y[i] = acc;
    }
  }

  /// Densify into a column-major Eigen matrix (repeated entries summed).
  ///
  /// Returns:
  ///     Eigen::MatrixXd: The `m x m` Laplacian.
  [[nodiscard]] Eigen::MatrixXd dense() const {
    const auto m = static_cast<Eigen::Index>(diag.size());
    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(m, m);
    for (Eigen::Index i = 0; i < m; ++i) {
      const auto ui = static_cast<std::size_t>(i);
      out(i, i) += diag[ui];
      for (std::size_t k = ptr[ui]; k < ptr[ui + 1]; ++k) {
        out(i, static_cast<Eigen::Index>(col[k])) += val[k];
      }
    }
    return out;
  }
};

/// The Laplacian deflated off the constant vector: `x -> P L P x + s mean(x) 1`, `P = I - 1 1^T/m`.
///
/// The constant vector spans the null space of a connected graph's Laplacian, so its eigenvalue 0
/// is the smallest, and a Lanczos solve asked for the smallest pair finds it as soon as rounding
/// gives the Krylov space a component along it, whatever the start vector. The projection removes
/// that component after every product; the shift `s` moves the constant direction to eigenvalue
/// `s`, so it is not the smallest eigenpair of the operator even in floating point. On the
/// complement of the constant vector the operator is `L`, so its smallest eigenpair is the Fiedler
/// pair `(lambda_2, u_2)` exactly when `s > lambda_2`.
///
/// Why `s` = the largest row sum of `|L|` works: it bounds `lambda_max(L)` (Gershgorin), and
/// `lambda_2 <= m / (m - 1) * d_min` (the Rayleigh quotient of `e_i - 1/m`, which is orthogonal to
/// the constant vector, is `d_i / (1 - 1/m)`), while the row sum of vertex `i` is `2 d_i` for
/// non-negative weights, so `s >= 2 d_max > m / (m - 1) * d_min >= lambda_2` for `m > 2` and
/// `d_min > 0`, which a connected graph with positive weights has. A connected graph whose stored
/// weights are all zero gives `s = 0` and a zero operator, on which Spectra throws; the caller then
/// takes the dense route.
///
/// Attributes:
///     lap (const Laplacian*): The Laplacian.
///     shift (double): The shift `s`.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
///     For general use, call `Spectral::partition` instead.
class DeflatedLaplacianOp {
 public:
  using Scalar = double;

  /// Wrap a Laplacian and compute its shift.
  ///
  /// Args:
  ///     lap (const Laplacian&): The Laplacian; must outlive the operator.
  explicit DeflatedLaplacianOp(const Laplacian& lap) : lap_(&lap), work_(lap.diag.size()) {
    for (std::size_t i = 0; i < lap.diag.size(); ++i) {
      double row = std::abs(lap.diag[i]);
      for (std::size_t k = lap.ptr[i]; k < lap.ptr[i + 1]; ++k) {
        row += std::abs(lap.val[k]);
      }
      shift_ = (std::max)(shift_, row);
    }
  }

  /// Get the operator's size.
  ///
  /// Returns:
  ///     Eigen::Index: The number of vertices.
  [[nodiscard]] Eigen::Index rows() const noexcept {
    return static_cast<Eigen::Index>(lap_->diag.size());
  }

  /// Get the operator's size.
  ///
  /// Returns:
  ///     Eigen::Index: The number of vertices.
  [[nodiscard]] Eigen::Index cols() const noexcept { return rows(); }

  /// Get the shift that moves the constant direction above the Fiedler value.
  ///
  /// Returns:
  ///     double: `s`.
  [[nodiscard]] double shift() const noexcept { return shift_; }

  /// Compute `y = P L P x + s mean(x) 1` (Spectra's operator interface).
  ///
  /// Args:
  ///     x_in (const double*): `rows()` entries.
  ///     y_out (double*): `rows()` entries, overwritten.
  void perform_op(const double* x_in, double* y_out) const {
    const std::size_t m = lap_->diag.size();
    const double mean_x = mean(x_in, m);
    for (std::size_t i = 0; i < m; ++i) {
      work_[i] = x_in[i] - mean_x;
    }
    lap_->apply(work_.data(), y_out);
    const double mean_y = mean(y_out, m);
    for (std::size_t i = 0; i < m; ++i) {
      y_out[i] = (y_out[i] - mean_y) + shift_ * mean_x;
    }
  }

 private:
  /// Get the mean of `m` values, summed sequentially.
  ///
  /// Args:
  ///     x (const double*): The values.
  ///     m (std::size_t): Their number, at least 1.
  ///
  /// Returns:
  ///     double: The mean.
  [[nodiscard]] static double mean(const double* x, std::size_t m) noexcept {
    double s = 0.0;
    for (std::size_t i = 0; i < m; ++i) {
      s += x[i];
    }
    return s / static_cast<double>(m);
  }

  const Laplacian* lap_;
  mutable std::vector<double> work_;
  double shift_ = 0.0;
};

/// Fill a fixed start vector for the Lanczos solve, uniform in `[-0.5, 0.5)`.
///
/// A fixed sequence (splitmix64) makes the solve deterministic on every platform; a generic vector
/// has a component along every eigenvector, which a structured one (an index ramp) can lack: the
/// centred ramp `i - (m - 1)/2` is orthogonal to every eigenvector symmetric under reversing the
/// vertex order.
///
/// Args:
///     m (std::size_t): The length.
///
/// Returns:
///     Eigen::VectorXd: The vector.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
///     For general use, call `Spectral::partition` instead.
[[nodiscard]] inline Eigen::VectorXd lanczos_start(std::size_t m) {
  Eigen::VectorXd v(static_cast<Eigen::Index>(m));
  std::uint64_t state = 0x9E3779B97F4A7C15ULL;
  for (Eigen::Index i = 0; i < v.size(); ++i) {
    state += 0x9E3779B97F4A7C15ULL;
    std::uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    // The top 53 bits as a double in [0, 1), exactly.
    v(i) = static_cast<double>(z >> 11) * 0x1.0p-53 - 0.5;
  }
  return v;
}

/// Solve `(T - sigma I) y = x` in place for a symmetric tridiagonal `T`, by Gaussian elimination
/// with partial pivoting (LAPACK `dgttrf` / `dgttrs`).
///
/// A pivot smaller in magnitude than `tiny` is replaced by `tiny` with its sign (`+tiny` for an
/// exact zero), as inverse iteration does (LAPACK `dlagtf`): `sigma` is an eigenvalue, so the
/// shifted matrix is singular to rounding, and the large solution is the point.
///
/// Args:
///     diag (const Eigen::VectorXd&): `T`'s diagonal, `n >= 2` entries.
///     off (const Eigen::VectorXd&): `T`'s off-diagonal, `n - 1` entries.
///     sigma (double): The shift.
///     tiny (double): The smallest pivot magnitude allowed, positive.
///     x (Eigen::VectorXd&): The right-hand side on entry, the solution on return.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
///     For general use, call `Spectral::partition` instead.
inline void shifted_tridiagonal_solve(const Eigen::VectorXd& diag, const Eigen::VectorXd& off,
                                      double sigma, double tiny, Eigen::VectorXd& x) {
  const Eigen::Index n = diag.size();
  Eigen::VectorXd d = diag.array() - sigma;
  Eigen::VectorXd du = off;                        // first superdiagonal of U
  Eigen::VectorXd dl = off;                        // multipliers
  Eigen::VectorXd du2 = Eigen::VectorXd::Zero(n);  // second superdiagonal of U
  std::vector<bool> swapped(static_cast<std::size_t>(n), false);
  auto guard = [tiny](double p) {
    if (std::abs(p) >= tiny) {
      return p;
    }
    return p < 0.0 ? -tiny : tiny;
  };
  for (Eigen::Index i = 0; i + 1 < n; ++i) {
    const double below = dl(i);
    if (std::abs(d(i)) >= std::abs(below)) {
      d(i) = guard(d(i));
      const double l = below / d(i);
      dl(i) = l;
      d(i + 1) -= l * du(i);
    } else {
      // Swap rows i and i + 1.
      swapped[static_cast<std::size_t>(i)] = true;
      const double l = d(i) / below;
      d(i) = below;
      const double t = d(i + 1);
      d(i + 1) = du(i) - l * t;
      if (i + 2 < n) {
        du2(i) = du(i + 1);
        du(i + 1) = -l * du(i + 1);
      }
      du(i) = t;
      dl(i) = l;
    }
  }
  d(n - 1) = guard(d(n - 1));
  // Forward: apply the row swaps and L^-1.
  for (Eigen::Index i = 0; i + 1 < n; ++i) {
    if (swapped[static_cast<std::size_t>(i)]) {
      const double t = x(i);
      x(i) = x(i + 1);
      x(i + 1) = t - dl(i) * x(i);
    } else {
      x(i + 1) -= dl(i) * x(i);
    }
  }
  // Backward: U, with two superdiagonals.
  x(n - 1) /= d(n - 1);
  if (n > 1) {
    x(n - 2) = (x(n - 2) - du(n - 2) * x(n - 1)) / d(n - 2);
  }
  for (Eigen::Index i = n - 3; i >= 0; --i) {
    x(i) = (x(i) - du(i) * x(i + 1) - du2(i) * x(i + 2)) / d(i);
  }
}

/// Compute the Fiedler vector of a connected subgraph's Laplacian by the dense route.
///
/// The Laplacian is reduced to tridiagonal form `T = Q^T L Q` (Householder, the lower triangle),
/// `T`'s eigenvalues are computed (implicit QR on `T` scaled to unit magnitude, values only), and
/// the eigenvector of the second smallest, `lambda_2`, by inverse iteration on `T - lambda_2 I`
/// with the constant vector's image `Q^T 1 / sqrt(m)` (the `lambda_1 = 0` eigenvector) projected
/// out after every solve, then mapped back by `Q`. This is the second column
/// `SelfAdjointEigenSolver` would return, up to sign and, when `lambda_2` is repeated, up to the
/// choice within its eigenspace, at a fraction of the cost: accumulating every eigenvector is cubic
/// with a large constant, this is one reduction and quadratic work after it.
///
/// Three solves: the first from a generic start leaves the other eigencomponents at
/// `|lambda_2 - sigma| / gap`, about `eps ||L|| / gap` relative, since the computed `sigma` is
/// within a few `eps ||L||` of `lambda_2`; the next two only refine. With the gap to `lambda_3`
/// at rounding level the vector is a mixture of the two, which is as determined as the problem
/// is (any vector of a near-repeated eigenspace orders the vertices equally well).
///
/// Args:
///     lap (const Laplacian&): The Laplacian, at least 2 vertices.
///
/// Returns:
///     Eigen::VectorXd: A unit Fiedler vector.
///
/// Raises:
///     state_error: The tridiagonal QR iteration does not converge within Eigen's limit (30
///         sweeps per eigenvalue), which a converged run of Eigen's dense solver also needs.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed).
///     For general use, call `Spectral::partition` instead.
[[nodiscard]] inline Eigen::VectorXd dense_fiedler(const Laplacian& lap) {
  const Eigen::MatrixXd a = lap.dense();
  const Eigen::Index m = a.rows();
  const Eigen::Tridiagonalization<Eigen::MatrixXd> tri(a);
  const Eigen::VectorXd diag = tri.diagonal();
  const Eigen::VectorXd off = tri.subDiagonal();
  const double norm =
      (std::max)(diag.cwiseAbs().maxCoeff(), off.size() > 0 ? off.cwiseAbs().maxCoeff() : 0.0);
  // Eigen's tridiagonal QR deflates an off-diagonal entry once `(e / eps)^2 <= |d_i| + |d_i+1|`,
  // a test that is not scale invariant and assumes entries of order one: `compute()` divides the
  // matrix by its largest entry first, and `computeFromTridiagonal` does not. Unscaled, a cluster
  // of equal eigenvalues (the clique Laplacian's 30, three times) keeps off-diagonal entries at
  // rounding level above that threshold, and whether the iteration converges depends on the last
  // bits (it did not in a build without FMA). Scaled as `compute()` scales, the test is the one
  // Eigen's dense solver runs.
  const double scale = norm > 0.0 ? norm : 1.0;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> values;
  values.computeFromTridiagonal(diag / scale, off / scale, Eigen::EigenvaluesOnly);
  if (values.info() != Eigen::Success) {
    throw state_error("partition_graph: the dense eigensolver did not converge on a subgraph of " +
                      std::to_string(m) + " vertices.");
  }
  const double sigma = values.eigenvalues()(1) * scale;
  // `eps ||T||`, LAPACK's perturbation of a small pivot in inverse iteration; a zero matrix (all
  // weights zero) takes 1, since any vector is then an eigenvector.
  const double tiny = std::numeric_limits<double>::epsilon() * (norm > 0.0 ? norm : 1.0);

  const Eigen::VectorXd ones =
      Eigen::VectorXd::Constant(m, 1.0 / std::sqrt(static_cast<double>(m)));
  const Eigen::VectorXd null_t = tri.matrixQ().transpose() * ones;
  Eigen::VectorXd x = lanczos_start(static_cast<std::size_t>(m));
  for (int solve = 0; solve < 3; ++solve) {
    x -= null_t.dot(x) * null_t;
    x /= x.norm();
    shifted_tridiagonal_solve(diag, off, sigma, tiny, x);
  }
  x -= null_t.dot(x) * null_t;
  x /= x.norm();
  return tri.matrixQ() * x;
}

}  // namespace detail

/// Recursive spectral (Fiedler) bisection.
///
/// Each step splits the part count `k` into `k / 2` and `k - k / 2`, orders the subgraph's
/// vertices by its Fiedler vector (the eigenvector of the second-smallest Laplacian eigenvalue),
/// and cuts at the weighted split point, clamped so that each half keeps at least as many vertices
/// as it has parts (no part is left empty). A disconnected subgraph is ordered by component
/// (components numbered by their smallest vertex, the subgraph's order kept within each), which
/// keeps each component whole. Connectivity counts every stored entry, a zero weight included.
///
/// A subgraph of at most `dense_max` vertices takes the dense route (`detail::dense_fiedler`, on
/// the lower triangle of the Laplacian); a larger one a Spectra Lanczos solve for the one smallest
/// eigenpair of the Laplacian deflated off the constant vector (`detail::DeflatedLaplacianOp`),
/// with `ncv = min(m, 20)`, at most `lanczos_max_iter` restarts (`10 m` when unset) and Spectra's
/// relative tolerance at machine epsilon (the defaults of scipy's `eigsh`). A solve that does not
/// converge, or that Spectra abandons with an exception, falls back to the dense route, counted in
/// `PartitionInfo::dense_fallbacks`.
///
/// The Laplacian is that of the stored weights, which the deflation argument assumes symmetric and
/// non-negative, as `coupling_graph` builds them; a hand-built graph that is neither is
/// partitioned without error, by an ordering with no Fiedler meaning.
///
/// Attributes:
///     dense_max (std::int64_t): The largest subgraph the dense solver takes by size.
///     lanczos_max_iter (std::optional<std::int64_t>): The Lanczos restart limit, at least 1;
///         `10 m` when unset.
struct Spectral {
  std::int64_t dense_max = dense_fiedler_max;
  std::optional<std::int64_t> lanczos_max_iter;

  /// Partition a graph (see `partition_graph`).
  ///
  /// Args:
  ///     graph (const CouplingGraph&): The graph.
  ///     n_parts (std::int64_t): Number of parts, at least 1.
  ///     active (std::optional<std::span<const std::uint8_t>>): One flag per vertex, or none.
  ///
  /// Returns:
  ///     GraphPartition: The partition and the solver counts.
  ///
  /// Raises:
  ///     value_error: As `partition_graph`, or `lanczos_max_iter < 1`.
  ///     state_error: The dense eigensolver does not converge (see `detail::dense_fiedler`).
  [[nodiscard]] GraphPartition partition(
      const CouplingGraph& graph, std::int64_t n_parts,
      std::optional<std::span<const std::uint8_t>> active = std::nullopt) const {
    if (lanczos_max_iter.has_value() && *lanczos_max_iter < 1) {
      throw value_error("lanczos_max_iter must be >= 1; got " + std::to_string(*lanczos_max_iter) +
                        ".");
    }
    const detail::ActiveSubgraph sub = detail::select_active(graph, n_parts, active);
    const std::size_t n_active = sub.active_idx.size();
    PartitionInfo info;
    std::vector<std::int32_t> owner(n_active, 0);
    std::vector<std::int64_t> local_of(n_active, -1);
    std::vector<double> cumw;

    auto bisect = [&](auto&& self, std::vector<std::int64_t> idx, std::int64_t part_lo,
                      std::int64_t part_hi) -> void {
      const std::int64_t k = part_hi - part_lo;
      if (k == 1) {
        for (const std::int64_t v : idx) {
          owner[static_cast<std::size_t>(v)] = grid::detail::to_owner(part_lo);
        }
        return;
      }
      std::vector<std::int64_t> ordered = order(sub, idx, local_of, info);
      cumw.resize(ordered.size());
      double running = 0.0;
      for (std::size_t i = 0; i < ordered.size(); ++i) {
        const double w = sub.vertex_weights[static_cast<std::size_t>(ordered[i])];
        running = i == 0 ? w : running + w;
        cumw[i] = running;
      }
      const auto split = static_cast<std::ptrdiff_t>(grid::detail::bisection_split(cumw, k));
      const std::int64_t k_left = k / 2;
      self(self, std::vector<std::int64_t>(ordered.begin(), ordered.begin() + split), part_lo,
           part_lo + k_left);
      self(self, std::vector<std::int64_t>(ordered.begin() + split, ordered.end()),
           part_lo + k_left, part_hi);
    };
    std::vector<std::int64_t> all(n_active);
    std::iota(all.begin(), all.end(), std::int64_t{0});
    bisect(bisect, std::move(all), std::int64_t{0}, n_parts);
    return {detail::scatter_owners(graph.num_vertices(), sub.active_idx, owner, n_parts), info};
  }

 private:
  /// Order a subgraph's vertices by its Fiedler vector, or by component when disconnected.
  ///
  /// Args:
  ///     sub (const detail::ActiveSubgraph&): The active subgraph.
  ///     idx (const std::vector<std::int64_t>&): The subgraph's vertices (at least 2), in the
  ///         order that numbers them locally.
  ///     local_of (std::vector<std::int64_t>&): Scratch, -1 everywhere on entry and on return.
  ///     info (PartitionInfo&): Updated with the path taken.
  ///
  /// Returns:
  ///     std::vector<std::int64_t>: `idx`, permuted.
  [[nodiscard]] std::vector<std::int64_t> order(const detail::ActiveSubgraph& sub,
                                                const std::vector<std::int64_t>& idx,
                                                std::vector<std::int64_t>& local_of,
                                                PartitionInfo& info) const {
    const std::size_t m = idx.size();
    for (std::size_t l = 0; l < m; ++l) {
      local_of[static_cast<std::size_t>(idx[l])] = static_cast<std::int64_t>(l);
    }
    // The edges among `idx`, in local numbering; every stored entry joins its ends.
    std::vector<std::size_t> parent(m);
    std::iota(parent.begin(), parent.end(), std::size_t{0});
    auto find = [&](std::size_t x) {
      while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
      }
      return x;
    };
    detail::Laplacian lap;
    lap.ptr.assign(m + 1, 0);
    lap.diag.assign(m, 0.0);
    for (std::size_t l = 0; l < m; ++l) {
      const auto v = static_cast<std::size_t>(idx[l]);
      for (auto k = static_cast<std::size_t>(sub.xadj[v]);
           k < static_cast<std::size_t>(sub.xadj[v + 1]); ++k) {
        const std::int64_t lc = local_of[static_cast<std::size_t>(sub.adjncy[k])];
        if (lc < 0) {
          continue;
        }
        const auto c = static_cast<std::size_t>(lc);
        const std::size_t ra = find(l);
        const std::size_t rb = find(c);
        if (ra != rb) {
          parent[(std::max)(ra, rb)] = (std::min)(ra, rb);
        }
        if (c != l) {
          const auto w = static_cast<double>(sub.edge_weights[k]);
          lap.col.push_back(c);
          lap.val.push_back(-w);
          lap.diag[c] += w;
        }
      }
      lap.ptr[l + 1] = lap.col.size();
    }
    for (const std::int64_t v : idx) {
      local_of[static_cast<std::size_t>(v)] = -1;
    }

    // Components numbered in the order of their smallest local vertex (scipy's labels).
    std::vector<std::size_t> label(m);
    std::vector<std::size_t> label_of_root(m, m);
    std::size_t n_components = 0;
    for (std::size_t l = 0; l < m; ++l) {
      const std::size_t r = find(l);
      if (label_of_root[r] == m) {
        label_of_root[r] = n_components++;
      }
      label[l] = label_of_root[r];
    }
    std::vector<std::size_t> perm(m);
    std::iota(perm.begin(), perm.end(), std::size_t{0});
    if (n_components > 1) {
      ++info.component_orders;
      std::stable_sort(perm.begin(), perm.end(),
                       [&](std::size_t a, std::size_t b) { return label[a] < label[b]; });
    } else {
      const Eigen::VectorXd fiedler = fiedler_vector(lap, info);
      Eigen::Index at = 0;
      for (Eigen::Index i = 1; i < fiedler.size(); ++i) {
        if (std::abs(fiedler(i)) > std::abs(fiedler(at))) {
          at = i;
        }
      }
      // An eigenvector is not zero, so its largest-magnitude entry is not either.
      const double sign = fiedler(at) > 0.0 ? 1.0 : -1.0;
      std::stable_sort(perm.begin(), perm.end(), [&](std::size_t a, std::size_t b) {
        return sign * fiedler(static_cast<Eigen::Index>(a)) <
               sign * fiedler(static_cast<Eigen::Index>(b));
      });
    }
    std::vector<std::int64_t> out(m);
    for (std::size_t i = 0; i < m; ++i) {
      out[i] = idx[perm[i]];
    }
    return out;
  }

  /// Compute the Fiedler vector of a connected subgraph's Laplacian.
  ///
  /// Args:
  ///     lap (const detail::Laplacian&): The Laplacian, at least 2 vertices.
  ///     info (PartitionInfo&): Updated with the solver used.
  ///
  /// Returns:
  ///     Eigen::VectorXd: A unit eigenvector of the second-smallest eigenvalue.
  [[nodiscard]] Eigen::VectorXd fiedler_vector(const detail::Laplacian& lap,
                                               PartitionInfo& info) const {
    const std::size_t m = lap.diag.size();
    if (static_cast<std::int64_t>(m) > dense_max) {
      detail::DeflatedLaplacianOp op(lap);
      const auto ncv = static_cast<Eigen::Index>((std::min)(m, std::size_t{20}));
      Spectra::SymEigsSolver<detail::DeflatedLaplacianOp> eigs(op, 1, ncv);
      const Eigen::VectorXd start = detail::lanczos_start(m);
      eigs.init(start.data());
      const auto maxit =
          static_cast<Eigen::Index>(lanczos_max_iter.value_or(10 * static_cast<std::int64_t>(m)));
      bool converged = false;
      try {
        static_cast<void>(eigs.compute(Spectra::SortRule::SmallestAlge, maxit,
                                       std::numeric_limits<double>::epsilon()));
        converged = eigs.info() == Spectra::CompInfo::Successful;
      } catch (const std::runtime_error&) {
        // Spectra throws when its small eigenproblem fails, which a zero operator causes: a
        // connected graph whose stored weights are all zero (ARPACK refuses that input as well).
        // The dense route answers every input.
        converged = false;
      }
      if (converged) {
        ++info.lanczos_solves;
        return eigs.eigenvectors().col(0);
      }
      ++info.dense_fallbacks;
    } else {
      ++info.dense_solves;
    }
    return detail::dense_fiedler(lap);
  }
};

static_assert(GraphPartitioner<Spectral>);

/// Partition a cell-coupling graph into `n_parts` rank subdomains.
///
/// Balances the vertex weights across parts while keeping the weight of cut edges (shared DOFs)
/// low. Vertices excluded by `active` get owner -1 and are dropped from the graph.
///
/// Args:
///     graph (const CouplingGraph&): The graph.
///     n_parts (std::int64_t): Number of parts, at least 1.
///     active (std::optional<std::span<const std::uint8_t>>): One flag per vertex (nonzero is
///         active), or no value for all active.
///     partitioner (const P&): The partitioner; `Spectral` by default.
///
/// Returns:
///     GraphPartition: The partition, `n_parts` parts, and how it was computed.
///
/// Raises:
///     value_error: `n_parts < 1`; `active` has the wrong length or marks no vertex; or `n_parts`
///         exceeds the number of active vertices.
template <GraphPartitioner P = Spectral>
[[nodiscard]] GraphPartition partition_graph(
    const CouplingGraph& graph, std::int64_t n_parts,
    std::optional<std::span<const std::uint8_t>> active = std::nullopt,
    const P& partitioner = P{}) {
  return partitioner.partition(graph, n_parts, active);
}

}  // namespace pantr::partition
