/// @file spanwise_extraction.hpp
/// @brief `SpanwiseElementExtraction`: the change of basis from a tensor-product B-spline space to
///        the Bézier, Lagrange or cardinal basis of each of its elements, as per-direction
///        operators applied matrix-free.
///
/// For a space of dimension `d` and a target basis, the operator of an element is the Kronecker
/// product `M = kron(M_0, ..., M_{d-1})` of the 1D extraction operators (`extraction.hpp`) of the
/// element's interval in each direction. It is never formed unless asked for (`kron_operator`):
/// `M v`, `M^T v`, `M^T K M` and `M K M^T` are mode products (`detail/kron_apply.hpp`), one per
/// direction and side, and a direction whose operator is structurally the identity is skipped.
///
/// The per-direction operators are stored compactly: only the non-identity ones, with a map from
/// each interval to its row. A one-cell application takes the cell as a per-direction index; a
/// batch takes `(n_cells, dim)` indices and runs the cells in parallel, each into its own output.
#pragma once

#include "pantr/basis/lagrange_variant.hpp"
#include "pantr/bspline/detail/kron_apply.hpp"
#include "pantr/bspline/extraction.hpp"
#include "pantr/bspline/space.hpp"
#include "pantr/bspline/space_1d.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/format.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"
#include "pantr/parallel/parallel_for.hpp"
#include "pantr/parallel/work_model.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace pantr::bspline {

/// The basis of each element that the operators map to (the source is always the B-spline basis).
///
/// The values are the wire values the Python binding passes.
enum class ExtractionTarget : int {
  bezier = 0,    ///< The Bernstein (Bézier) basis.
  lagrange = 1,  ///< The Lagrange basis at the nodes of a `LagrangeVariant`.
  cardinal = 2,  ///< The cardinal B-spline basis.
};

/// Convert a wire value to an `ExtractionTarget`.
///
/// Raises:
///     value_error: `value` names no target.
[[nodiscard]] inline ExtractionTarget to_extraction_target(int value) {
  if (value < 0 || value > 2) {
    throw value_error("no extraction target has the value " + std::to_string(value));
  }
  return static_cast<ExtractionTarget>(value);
}

namespace detail {

/// The refusal of a periodic direction, spelled as the Python class has always spelled it.
inline constexpr const char* periodic_not_supported =
    "SpanwiseElementExtraction does not yet support periodic directions. "
    "Convert the B-spline to open form first (see Bspline.to_open_bspline).";

/// One direction of a `SpanwiseElementExtraction`: its operators and which intervals they cover.
///
/// Attributes:
///     compact (NdArray<T>): `(n_compact, n_out, n_in)`: the operators of the non-identity
///         intervals, or one zero operator when every interval is identity.
///     idx_map (std::vector<std::int64_t>): Per interval, its row of `compact` (0 for an identity
///         interval, which is never read).
///     mask (std::unique_ptr<bool[]>): Per interval, whether its operator is the identity.
///     n_intervals (std::size_t): The intervals.
///     n_identity (std::size_t): How many of them are identity.
template <StorageFloat T>
struct ExtractionDirection {
  NdArray<T> compact;
  std::vector<std::int64_t> idx_map;
  std::unique_ptr<bool[]> mask;
  std::size_t n_intervals;
  std::size_t n_identity;
};

/// The shared representation of a `SpanwiseElementExtraction`.
template <StorageFloat T>
struct SpanwiseRep {
  BsplineSpace<T> space;
  ExtractionTarget target;
  basis::LagrangeVariant variant;
  std::vector<ExtractionDirection<T>> directions;
  std::vector<std::size_t> num_intervals;
  std::vector<std::size_t> rows;  // n_out per direction
  std::vector<std::size_t> cols;  // n_in per direction
  std::array<KronPlan, 4> plans;  // by ExtractionOp
  std::size_t total_intervals;
};

/// Build one direction: the operators of `target`, the identity mask, and the compact form.
template <StorageFloat T>
[[nodiscard]] ExtractionDirection<T> make_direction(const BsplineSpace1D<T>& space,
                                                    ExtractionTarget target,
                                                    basis::LagrangeVariant variant) {
  const auto n_el = static_cast<std::size_t>(space.num_intervals());
  const auto order = static_cast<std::size_t>(space.degree()) + 1;
  std::unique_ptr<bool[]> mask(new bool[n_el]);
  const std::span<bool> flags(mask.get(), n_el);
  NdArray<T> ops({n_el, order, order}, for_overwrite);
  switch (target) {
    case ExtractionTarget::bezier:
      bezier_extraction_operators<T>(space, ops.template view<3>());
      bezier_structural_identity_mask<T>(space, flags);
      break;
    case ExtractionTarget::lagrange:
      lagrange_extraction_operators<T>(space, variant, ops.template view<3>());
      lagrange_structural_identity_mask<T>(space, variant, flags);
      break;
    case ExtractionTarget::cardinal:
      cardinal_extraction_operators<T>(space, ops.template view<3>());
      space.cardinal_intervals(flags);
      break;
  }
  const auto n_identity = static_cast<std::size_t>(std::count(flags.begin(), flags.end(), true));
  const std::size_t n_compact = n_el - n_identity;
  // At least one row, so the array is never empty: a direction of identity intervals keeps a zero
  // operator that nothing reads.
  NdArray<T> compact({(std::max)(n_compact, std::size_t{1}), order, order});
  std::vector<std::int64_t> idx_map(n_el, 0);
  std::size_t row = 0;
  for (std::size_t e = 0; e < n_el; ++e) {
    if (!flags[e]) {
      std::copy_n(ops.data() + e * order * order, order * order,
                  compact.data() + row * order * order);
      idx_map[e] = static_cast<std::int64_t>(row);
      ++row;
    }
  }
  return ExtractionDirection<T>{std::move(compact), std::move(idx_map), std::move(mask), n_el,
                                n_identity};
}

}  // namespace detail

/// The element operators of a tensor-product B-spline space onto a target basis.
///
/// An immutable value (design §2.2): the operators are built once, at construction, and shared,
/// read-only, by every copy; every application reads them and writes only the caller's buffers, so
/// the applications of one value may run concurrently. With `d` directions, `M = kron(M_0, ...,
/// M_{d-1})` is the operator of an element, `M_k` the 1D operator of its interval in direction `k`
/// (`n_out_k x n_in_k`; every target builds square ones, `degree_k + 1` on a side, and the class
/// is written for others). The cell of a one-cell application is the per-direction interval index;
/// the elements of a batch are the rows of a `(n_cells, d)` array of such indices.
///
/// Attributes:
///     rep_ (std::shared_ptr<const detail::SpanwiseRep<T>>): The shared representation.
template <StorageFloat T>
class SpanwiseElementExtraction {
 public:
  /// The shared representation type.
  using Rep = detail::SpanwiseRep<T>;

  /// Build the operators of every direction.
  ///
  /// Args:
  ///     space (const BsplineSpace<T>&): The space; no direction may be periodic.
  ///     target (ExtractionTarget): The basis of each element.
  ///     variant (basis::LagrangeVariant): The Lagrange node distribution, used by
  ///         `ExtractionTarget::lagrange` only. Defaults to equispaced.
  ///
  /// Raises:
  ///     not_implemented: A direction is periodic.
  ///     value_error: The degree of a direction is beyond the degrees the target's change-of-basis
  ///         matrix can be built for.
  SpanwiseElementExtraction(const BsplineSpace<T>& space, ExtractionTarget target,
                            basis::LagrangeVariant variant = basis::LagrangeVariant::equispaces)
      : rep_(make_rep(space, target, variant)) {}

  /// Get the space the value was built from.
  [[nodiscard]] BsplineSpace<T> space() const { return rep_->space; }
  /// Get the target basis.
  [[nodiscard]] ExtractionTarget target() const noexcept { return rep_->target; }
  /// Get the Lagrange node distribution (meaningful for the Lagrange target only).
  [[nodiscard]] basis::LagrangeVariant lagrange_variant() const noexcept { return rep_->variant; }
  /// Get the number of directions.
  [[nodiscard]] std::size_t dim() const noexcept { return rep_->directions.size(); }
  /// Get the intervals per direction.
  [[nodiscard]] std::span<const std::size_t> num_intervals() const noexcept {
    return rep_->num_intervals;
  }
  /// Get the number of elements, the product of the intervals per direction.
  [[nodiscard]] std::size_t num_total_intervals() const noexcept { return rep_->total_intervals; }
  /// Get the input size of the 1D operator in each direction (its columns).
  [[nodiscard]] std::span<const std::size_t> input_shape_per_dir() const noexcept {
    return rep_->cols;
  }
  /// Get the output size of the 1D operator in each direction (its rows).
  [[nodiscard]] std::span<const std::size_t> output_shape_per_dir() const noexcept {
    return rep_->rows;
  }

  /// Get the compact operators of a direction: `(n_compact, n_out, n_in)`, the non-identity
  /// intervals' operators in order, or one zero operator when every interval is identity.
  ///
  /// Args:
  ///     k (std::size_t): The direction.
  ///
  /// Raises:
  ///     index_error: `k` is not below `dim()`.
  [[nodiscard]] const NdArray<T>& compact_ops(std::size_t k) const { return direction(k).compact; }
  /// Get the row of `compact_ops(k)` of each interval of a direction (0 for an identity interval).
  ///
  /// Raises:
  ///     index_error: `k` is not below `dim()`.
  [[nodiscard]] std::span<const std::int64_t> idx_map(std::size_t k) const {
    return direction(k).idx_map;
  }
  /// Get, per interval of a direction, whether its operator is the identity.
  ///
  /// Raises:
  ///     index_error: `k` is not below `dim()`.
  [[nodiscard]] std::span<const bool> identity_mask(std::size_t k) const {
    const auto& dir = direction(k);
    return {dir.mask.get(), dir.n_intervals};
  }
  /// Get the number of elements whose operator is the identity in every direction.
  [[nodiscard]] std::size_t num_identity_elements() const noexcept {
    std::size_t n = 1;
    for (const auto& dir : rep_->directions) {
      n *= dir.n_identity;
    }
    return n;
  }
  /// Report whether every element's operator is the identity.
  [[nodiscard]] bool is_identity() const noexcept {
    return std::all_of(rep_->directions.begin(), rep_->directions.end(),
                       [](const auto& dir) { return dir.n_identity == dir.n_intervals; });
  }

  /// Get the scratch an application needs, in entries (see `required_scratch_size`).
  ///
  /// Raises:
  ///     value_error: `op` is not one of the operations.
  [[nodiscard]] std::size_t required_scratch(ExtractionOp op) const { return 2 * plan_of(op).half; }

  /// Get the operations one cell of a batch costs, an upper bound (a cell with identity
  /// directions does less), at least 1: what a caller weighs against the cost of waking the pool.
  ///
  /// Raises:
  ///     value_error: `op` is not one of the operations.
  [[nodiscard]] double work_per_cell(ExtractionOp op) const { return plan_of(op).ops; }

  /// Get the per-direction index of a flat element index (C order over `num_intervals()`).
  ///
  /// Raises:
  ///     index_error: `flat` is not below `num_total_intervals()`.
  [[nodiscard]] std::vector<std::size_t> unravel(std::size_t flat) const {
    if (flat >= rep_->total_intervals) {
      throw index_error("Flat cell index " + std::to_string(flat) + " out of range for " +
                        std::to_string(rep_->total_intervals) + " elements");
    }
    std::vector<std::size_t> cell(dim());
    for (std::size_t k = dim(); k-- > 0;) {
      cell[k] = flat % rep_->num_intervals[k];
      flat /= rep_->num_intervals[k];
    }
    return cell;
  }

  /// Compute `out = M v` for one element.
  ///
  /// Args:
  ///     v (std::span<const T>): `prod(input_shape_per_dir())` entries.
  ///     cell (std::span<const std::size_t>): The element's interval in each direction.
  ///     out (std::span<T>): `prod(output_shape_per_dir())` entries; overwritten.
  ///     scratch (std::optional<std::span<T>>): At least `required_scratch(ExtractionOp::apply)`
  ///         entries, or none to allocate.
  ///
  /// Raises:
  ///     value_error: `cell` has another length, an operand or `out` has another size, `scratch` is
  ///         smaller than needed, `out` overlaps `v` (unless every direction of the element is
  ///         identity) or `scratch` overlaps either.
  ///     index_error: An entry of `cell` is out of range.
  void apply(std::span<const T> v, std::span<const std::size_t> cell, std::span<T> out,
             std::optional<std::span<T>> scratch = std::nullopt) const {
    run_vector(ExtractionOp::apply, v, cell, out, scratch);
  }

  /// Compute `out = M^T v` for one element (see `apply`; `v` has `prod(output_shape_per_dir())`
  /// entries and `out` `prod(input_shape_per_dir())`).
  void apply_transpose(std::span<const T> v, std::span<const std::size_t> cell, std::span<T> out,
                       std::optional<std::span<T>> scratch = std::nullopt) const {
    run_vector(ExtractionOp::apply_transpose, v, cell, out, scratch);
  }

  /// Compute `out = M^T K M` for one element (see `apply`; `K` is `prod(output_shape_per_dir())`
  /// square and `out` `prod(input_shape_per_dir())` square).
  void apply_mt_k_m(mdview<const T, 2> k, std::span<const std::size_t> cell, mdview<T, 2> out,
                    std::optional<std::span<T>> scratch = std::nullopt) const {
    run_matrix(ExtractionOp::mt_k_m, k, cell, out, scratch);
  }

  /// Compute `out = M K M^T` for one element (see `apply`; `K` is `prod(input_shape_per_dir())`
  /// square and `out` `prod(output_shape_per_dir())` square).
  void apply_m_k_mt(mdview<const T, 2> k, std::span<const std::size_t> cell, mdview<T, 2> out,
                    std::optional<std::span<T>> scratch = std::nullopt) const {
    run_matrix(ExtractionOp::m_k_mt, k, cell, out, scratch);
  }

  /// Compute `out[c] = M_c v[c]` for every cell of a batch, in parallel over the cells.
  ///
  /// Args:
  ///     v (mdview<const T, 2>): `(n_cells, prod(input_shape_per_dir()))`.
  ///     cells (mdview<const std::int64_t, 2>): `(n_cells, dim())`: each row an element's interval
  ///         in each direction.
  ///     out (mdview<T, 2>): `(n_cells, prod(output_shape_per_dir()))`; overwritten.
  ///     scratch (std::optional<mdview<T, 2>>): `(n_cells, w)` with `w` at least
  ///         `max(required_scratch(ExtractionOp::apply), 1)`, a row per cell, or none (each chunk
  ///         of cells then allocates its own).
  ///
  /// Raises:
  ///     value_error: `cells` has another number of columns or rows than `v`, an operand or `out`
  ///         has another shape, `scratch` has another number of rows or is narrower than needed,
  ///         `out` overlaps `v` (unless every cell is identity in every direction) or `scratch`
  ///         overlaps either.
  ///     index_error: An entry of `cells` is out of range for its direction.
  void apply_many(mdview<const T, 2> v, mdview<const std::int64_t, 2> cells, mdview<T, 2> out,
                  std::optional<mdview<T, 2>> scratch = std::nullopt) const {
    run_many<2>(ExtractionOp::apply, v.data_handle(), {v.extent(0), v.extent(1)}, cells,
                out.data_handle(), {out.extent(0), out.extent(1)}, scratch);
  }

  /// Compute `out[c] = M_c^T v[c]` for every cell of a batch (see `apply_many`; `v` is `(n_cells,
  /// prod(output_shape_per_dir()))` and `out` `(n_cells, prod(input_shape_per_dir()))`).
  void apply_transpose_many(mdview<const T, 2> v, mdview<const std::int64_t, 2> cells,
                            mdview<T, 2> out,
                            std::optional<mdview<T, 2>> scratch = std::nullopt) const {
    run_many<2>(ExtractionOp::apply_transpose, v.data_handle(), {v.extent(0), v.extent(1)}, cells,
                out.data_handle(), {out.extent(0), out.extent(1)}, scratch);
  }

  /// Compute `out[c] = M_c^T K[c] M_c` for every cell of a batch (see `apply_many`; `K` is
  /// `(n_cells, N, N)`, `N = prod(output_shape_per_dir())`, and `out` `(n_cells, N', N')`, `N' =
  /// prod(input_shape_per_dir())`).
  void apply_mt_k_m_many(mdview<const T, 3> k, mdview<const std::int64_t, 2> cells,
                         mdview<T, 3> out,
                         std::optional<mdview<T, 2>> scratch = std::nullopt) const {
    run_many<3>(ExtractionOp::mt_k_m, k.data_handle(), {k.extent(0), k.extent(1), k.extent(2)},
                cells, out.data_handle(), {out.extent(0), out.extent(1), out.extent(2)}, scratch);
  }

  /// Compute `out[c] = M_c K[c] M_c^T` for every cell of a batch (see `apply_many`; `K` is
  /// `(n_cells, N, N)`, `N = prod(input_shape_per_dir())`, and `out` `(n_cells, N', N')`, `N' =
  /// prod(output_shape_per_dir())`).
  void apply_m_k_mt_many(mdview<const T, 3> k, mdview<const std::int64_t, 2> cells,
                         mdview<T, 3> out,
                         std::optional<mdview<T, 2>> scratch = std::nullopt) const {
    run_many<3>(ExtractionOp::m_k_mt, k.data_handle(), {k.extent(0), k.extent(1), k.extent(2)},
                cells, out.data_handle(), {out.extent(0), out.extent(1), out.extent(2)}, scratch);
  }

  /// Form the operator of one element, `kron(M_0, ..., M_{d-1})` (identity directions as the
  /// identity matrix), each entry the product of the factors' entries from the first direction to
  /// the last.
  ///
  /// Args:
  ///     cell (std::span<const std::size_t>): The element's interval in each direction.
  ///     out (mdview<T, 2>): `(prod(output_shape_per_dir()), prod(input_shape_per_dir()))`;
  ///         overwritten.
  ///
  /// Raises:
  ///     value_error: `cell` has another length or `out` another shape.
  ///     index_error: An entry of `cell` is out of range.
  void kron_operator(std::span<const std::size_t> cell, mdview<T, 2> out) const {
    check_cell(cell);
    const std::size_t n_rows = rep_->plans[0].result_size;
    const std::size_t n_cols = rep_->plans[0].operand_size;  // the plan of `M v`: N_in -> N_out
    basis::detail::check_out<T, 2>(out, {n_rows, n_cols});
    // Built direction by direction as `np.kron` does: `acc` is the product of the first `k`
    // factors, `a_rows x a_cols`, and the next one multiplies each of its entries.
    std::vector<T> acc(1, static_cast<T>(1));
    std::size_t a_rows = 1;
    std::size_t a_cols = 1;
    for (std::size_t k = 0; k < dim(); ++k) {
      const auto& dir = rep_->directions[k];
      const std::size_t r = rep_->rows[k];
      const std::size_t c = rep_->cols[k];
      const bool ident = dir.mask[cell[k]];
      const T* f = dir.compact.data() + static_cast<std::size_t>(dir.idx_map[cell[k]]) * r * c;
      const bool last = k + 1 == dim();
      std::vector<T> next;
      if (!last) {
        next.resize(a_rows * r * a_cols * c);
      }
      T* dst = last ? out.data_handle() : next.data();
      for (std::size_t i = 0; i < a_rows; ++i) {
        for (std::size_t p = 0; p < r; ++p) {
          for (std::size_t j = 0; j < a_cols; ++j) {
            for (std::size_t q = 0; q < c; ++q) {
              const T fv = ident ? (p == q ? static_cast<T>(1) : static_cast<T>(0)) : f[p * c + q];
              dst[(i * r + p) * (a_cols * c) + j * c + q] = acc[i * a_cols + j] * fv;
            }
          }
        }
      }
      a_rows *= r;
      a_cols *= c;
      acc = std::move(next);
    }
  }

  /// Get the shared representation, for an owner that must keep the arrays alive (a view handed to
  /// another language).
  [[nodiscard]] const std::shared_ptr<const Rep>& rep() const noexcept { return rep_; }

 private:
  static std::shared_ptr<const Rep> make_rep(const BsplineSpace<T>& space, ExtractionTarget target,
                                             basis::LagrangeVariant variant) {
    for (const auto& s : space.spaces()) {
      if (s.periodic()) {
        throw not_implemented(detail::periodic_not_supported);
      }
    }
    Rep rep{space, target, variant, {}, {}, {}, {}, {}, 1};
    for (const auto& s : space.spaces()) {
      rep.directions.push_back(detail::make_direction<T>(s, target, variant));
      const auto order = static_cast<std::size_t>(s.degree()) + 1;
      rep.num_intervals.push_back(static_cast<std::size_t>(s.num_intervals()));
      rep.rows.push_back(order);
      rep.cols.push_back(order);
    }
    rep.total_intervals = checked_product(rep.num_intervals);
    for (std::size_t op = 0; op < rep.plans.size(); ++op) {
      rep.plans[op] = detail::make_kron_plan(static_cast<ExtractionOp>(op), rep.rows, rep.cols);
    }
    return std::make_shared<const Rep>(std::move(rep));
  }

  /// Get the plan of an operation.
  ///
  /// Raises:
  ///     value_error: `op` is not one of the operations (a cast from an integer outside them).
  [[nodiscard]] const detail::KronPlan& plan_of(ExtractionOp op) const {
    const auto index = static_cast<std::size_t>(op);
    if (index >= rep_->plans.size()) {
      throw value_error("no extraction operation has the value " + std::to_string(index));
    }
    return rep_->plans[index];
  }

  /// Get a direction's data.
  ///
  /// Raises:
  ///     index_error: `k` is not below `dim()`.
  [[nodiscard]] const detail::ExtractionDirection<T>& direction(std::size_t k) const {
    if (k >= dim()) {
      throw index_error("direction " + std::to_string(k) + " out of range for " +
                        std::to_string(dim()) + " directions");
    }
    return rep_->directions[k];
  }

  /// Refuse a cell of another length than `dim()` or with an entry out of range.
  void check_cell(std::span<const std::size_t> cell) const {
    if (cell.size() != dim()) {
      throw value_error("Per-direction cell index has length " + std::to_string(cell.size()) +
                        ", expected " + std::to_string(dim()));
    }
    for (std::size_t k = 0; k < dim(); ++k) {
      if (cell[k] >= rep_->num_intervals[k]) {
        throw index_error("Cell index " + std::to_string(cell[k]) + " out of range for direction " +
                          std::to_string(k) + " with " + std::to_string(rep_->num_intervals[k]) +
                          " elements");
      }
    }
  }

  /// Refuse a vector of other size than `want`, with the message of a shape mismatch.
  static void check_vector(const char* what, std::size_t got, std::size_t want) {
    if (got != want) {
      const std::size_t g[1] = {got};
      const std::size_t w[1] = {want};
      throw value_error(std::string(what) + " has shape " +
                        py_repr(std::span<const std::size_t>(g)) + ", but expected shape " +
                        py_repr(std::span<const std::size_t>(w)));
    }
  }

  /// Refuse a scratch smaller than the operation needs, or overlapping the operand or the result.
  void check_scratch(ExtractionOp op, std::span<T> scratch, std::span<const T> operand,
                     std::span<T> out) const {
    const std::size_t need = required_scratch(op);
    if (scratch.size() < need) {
      throw value_error("Scratch size " + std::to_string(scratch.size()) +
                        " is smaller than required " + std::to_string(need));
    }
    if (detail::ranges_overlap<T>(scratch.data(), scratch.size(), operand.data(), operand.size()) ||
        detail::ranges_overlap<T>(scratch.data(), scratch.size(), out.data(), out.size())) {
      throw value_error("scratch must not overlap the operand or out");
    }
  }

  /// Run one cell given flat operand and result (the shapes are checked by the callers).
  void run_cell(ExtractionOp op, std::span<const T> operand, std::span<const std::size_t> cell,
                std::span<T> out, std::optional<std::span<T>> scratch) const {
    const detail::KronPlan& plan = plan_of(op);
    const std::size_t d = dim();
    std::vector<const T*> values(d);
    const std::unique_ptr<bool[]> ident(new bool[d]);
    bool all_identity = true;
    for (std::size_t k = 0; k < d; ++k) {
      const auto& dir = rep_->directions[k];
      values[k] = dir.compact.data() +
                  static_cast<std::size_t>(dir.idx_map[cell[k]]) * rep_->rows[k] * rep_->cols[k];
      ident[k] = dir.mask[cell[k]];
      all_identity = all_identity && ident[k];
    }
    // The stages read and write overlapping memory unless the cell is a plain copy.
    if (!all_identity &&
        detail::ranges_overlap<T>(out.data(), out.size(), operand.data(), operand.size())) {
      throw value_error(
          "out must not alias the operand; the kernel reads and writes overlapping memory");
    }
    std::unique_ptr<T[]> owned;
    T* work = nullptr;
    if (scratch.has_value()) {
      check_scratch(op, *scratch, operand, out);
      work = scratch->data();
    } else if (plan.half != 0) {
      owned.reset(new T[2 * plan.half]);
      work = owned.get();
    }
    detail::kron_execute<T>(plan, values.data(), ident.get(), operand.data(), out.data(), work);
  }

  void run_vector(ExtractionOp op, std::span<const T> v, std::span<const std::size_t> cell,
                  std::span<T> out, std::optional<std::span<T>> scratch) const {
    check_cell(cell);
    const detail::KronPlan& plan = plan_of(op);
    check_vector("Operand", v.size(), plan.operand_size);
    check_vector("Output array", out.size(), plan.result_size);
    run_cell(op, v, cell, out, scratch);
  }

  void run_matrix(ExtractionOp op, mdview<const T, 2> k, std::span<const std::size_t> cell,
                  mdview<T, 2> out, std::optional<std::span<T>> scratch) const {
    check_cell(cell);
    const detail::KronPlan& plan = plan_of(op);
    const std::size_t n_in = plan.operand_side;
    const std::size_t n_out = plan.result_side;
    if (k.extent(0) != n_in || k.extent(1) != n_in) {
      const std::size_t g[2] = {k.extent(0), k.extent(1)};
      const std::size_t w[2] = {n_in, n_in};
      throw value_error("Operand has shape " + py_repr(std::span<const std::size_t>(g)) +
                        ", but expected shape " + py_repr(std::span<const std::size_t>(w)));
    }
    basis::detail::check_out<T, 2>(out, {n_out, n_out});
    run_cell(op, std::span<const T>(k.data_handle(), plan.operand_size), cell,
             std::span<T>(out.data_handle(), plan.result_size), scratch);
  }

  /// Run a batch: `operand` and `out` are `n_cells` consecutive cells' arrays, described by their
  /// extents (the cell count first, then the shape of one cell's array).
  template <std::size_t R>
  void run_many(ExtractionOp op, const T* operand, std::array<std::size_t, R> operand_extents,
                mdview<const std::int64_t, 2> cells, T* out, std::array<std::size_t, R> out_extents,
                std::optional<mdview<T, 2>> scratch) const {
    const detail::KronPlan& plan = plan_of(op);
    const std::size_t d = dim();
    const std::size_t n_cells = operand_extents[0];
    if (cells.extent(1) != d) {
      throw value_error("cell_indices.shape[1]=" + std::to_string(cells.extent(1)) +
                        " does not match d=" + std::to_string(d));
    }
    // One cell's operand is a vector `(N,)` or a square matrix `(N, N)`; so is its result.
    std::array<std::size_t, R> want_operand{};
    std::array<std::size_t, R> want_out{};
    want_operand[0] = want_out[0] = cells.extent(0);
    for (std::size_t r = 1; r < R; ++r) {
      want_operand[r] = plan.operand_side;
      want_out[r] = plan.result_side;
    }
    if (operand_extents != want_operand) {
      throw value_error(
          "Operand has shape " + py_repr(std::span<const std::size_t>(operand_extents)) +
          ", but expected shape " + py_repr(std::span<const std::size_t>(want_operand)));
    }
    if (out_extents != want_out) {
      throw value_error("Output array has shape " +
                        py_repr(std::span<const std::size_t>(out_extents)) +
                        ", but expected shape " + py_repr(std::span<const std::size_t>(want_out)));
    }
    bool all_identity = true;
    for (std::size_t k = 0; k < d; ++k) {
      const auto& dir = rep_->directions[k];
      const auto n_el = static_cast<std::int64_t>(dir.n_intervals);
      for (std::size_t c = 0; c < n_cells; ++c) {
        const std::int64_t e = cells(c, k);
        if (e < 0 || e >= n_el) {
          throw index_error("cell_indices[:, " + std::to_string(k) +
                            "] contains values outside [0, " + std::to_string(dir.n_intervals) +
                            ")");
        }
        all_identity = all_identity && dir.mask[static_cast<std::size_t>(e)];
      }
    }
    const std::size_t operand_total = n_cells * plan.operand_size;
    const std::size_t out_total = n_cells * plan.result_size;
    if (!all_identity && detail::ranges_overlap<T>(out, out_total, operand, operand_total)) {
      throw value_error(
          "out must not alias the operand; the kernel reads and writes overlapping memory");
    }
    T* scratch_data = nullptr;
    std::size_t scratch_ld = 0;
    if (scratch.has_value()) {
      const std::size_t width = (std::max)(required_scratch(op), std::size_t{1});
      if (scratch->extent(0) != n_cells) {
        throw value_error("Batch scratch shape[0]=" + std::to_string(scratch->extent(0)) +
                          " does not match n_cells=" + std::to_string(n_cells));
      }
      if (scratch->extent(1) < width) {
        throw value_error("Batch scratch shape[1]=" + std::to_string(scratch->extent(1)) +
                          " is smaller than required width=" + std::to_string(width));
      }
      scratch_data = scratch->data_handle();
      scratch_ld = scratch->extent(1);
      const std::size_t scratch_total = n_cells * scratch_ld;
      if (detail::ranges_overlap<T>(scratch_data, scratch_total, operand, operand_total) ||
          detail::ranges_overlap<T>(scratch_data, scratch_total, out, out_total)) {
        throw value_error("scratch must not overlap the operand or out");
      }
    }
    // One cell costs the plan's multiply-adds (an upper bound: identity directions are skipped).
    // The chunks are cut by those operations alone: a cell does many multiply-adds on its small
    // operand and result, which stay in cache, so it is compute-bound and the bytes it writes do
    // not bound the chunk from below (the `bytes_per_item` overload is for cheap items that write
    // much).
    const parallel::WorkEstimate work{plan.ops};
    parallel::parallel_for(n_cells, work, [&](std::size_t begin, std::size_t end) {
      std::vector<const T*> values(d);
      const std::unique_ptr<bool[]> ident(new bool[d]);
      std::unique_ptr<T[]> own;
      if (scratch_data == nullptr && plan.half != 0) {
        own.reset(new T[2 * plan.half]);
      }
      for (std::size_t c = begin; c < end; ++c) {
        for (std::size_t k = 0; k < d; ++k) {
          const auto& dir = rep_->directions[k];
          const auto e = static_cast<std::size_t>(cells(c, k));
          values[k] = dir.compact.data() +
                      static_cast<std::size_t>(dir.idx_map[e]) * rep_->rows[k] * rep_->cols[k];
          ident[k] = dir.mask[e];
        }
        T* work_c = scratch_data != nullptr ? scratch_data + c * scratch_ld : own.get();
        detail::kron_execute<T>(plan, values.data(), ident.get(), operand + c * plan.operand_size,
                                out + c * plan.result_size, work_c);
      }
    });
  }

  std::shared_ptr<const Rep> rep_;
};

/// Apply a Kronecker product `kron(M_0, ..., M_{d-1})` of arbitrary factors to one operand.
///
/// The same computation as the class's, on factors that need not come from a space (any shapes; an
/// identity-flagged factor must be square). `operand` and `out` are the flat row-major vector or
/// square matrix the operation reads and writes.
///
/// Args:
///     op (ExtractionOp): The operation.
///     factors (std::span<const KronFactor<T>>): One per direction, at least one.
///     operand (std::span<const T>): The vector or matrix the operation reads.
///     out (std::span<T>): The vector or matrix it writes; overwritten.
///     scratch (std::span<T>): At least `required_scratch_size` entries.
///
/// Raises:
///     value_error: No factor; an identity-flagged factor that is not square; a factor with the
///         wrong number of values; an operand, `out` or `scratch` of the wrong size; `out`
///         overlaps `operand` (unless every factor is identity) or `scratch` overlaps either.
///     capacity_error: A product of extents overflows `std::size_t`.
template <StorageFloat T>
void apply_kron(ExtractionOp op, std::span<const KronFactor<T>> factors, std::span<const T> operand,
                std::span<T> out, std::span<T> scratch) {
  if (factors.empty()) {
    throw value_error("At least one direction is required");
  }
  std::vector<std::size_t> rows;
  std::vector<std::size_t> cols;
  std::vector<const T*> values;
  const std::unique_ptr<bool[]> ident(new bool[factors.size()]);
  bool all_identity = true;
  for (std::size_t k = 0; k < factors.size(); ++k) {
    const KronFactor<T>& f = factors[k];
    if (f.identity && f.rows != f.cols) {
      const std::size_t shape[2] = {f.rows, f.cols};
      throw value_error("factors[" + std::to_string(k) + "] is flagged identity but has shape " +
                        py_repr(std::span<const std::size_t>(shape)) +
                        "; an identity operator must be square");
    }
    // `rows * cols` is formed only when it fits: a product past size_t matches no span.
    const bool fits = f.rows == 0 || f.cols <= (std::numeric_limits<std::size_t>::max)() / f.rows;
    if (!f.identity && (!fits || f.values.size() != f.rows * f.cols)) {
      throw value_error("factors[" + std::to_string(k) + "] holds " +
                        std::to_string(f.values.size()) + " values, but its shape needs " +
                        std::to_string(f.rows) + " x " + std::to_string(f.cols));
    }
    rows.push_back(f.rows);
    cols.push_back(f.cols);
    values.push_back(f.values.data());
    ident[k] = f.identity;
    all_identity = all_identity && f.identity;
  }
  const detail::KronPlan plan = detail::make_kron_plan(op, rows, cols);
  if (operand.size() != plan.operand_size) {
    throw value_error("Operand has " + std::to_string(operand.size()) + " values, but expected " +
                      std::to_string(plan.operand_size));
  }
  if (out.size() != plan.result_size) {
    throw value_error("Output array has " + std::to_string(out.size()) + " values, but expected " +
                      std::to_string(plan.result_size));
  }
  if (scratch.size() < 2 * plan.half) {
    throw value_error("Scratch size " + std::to_string(scratch.size()) +
                      " is smaller than required " + std::to_string(2 * plan.half));
  }
  if (!all_identity &&
      detail::ranges_overlap<T>(out.data(), out.size(), operand.data(), operand.size())) {
    throw value_error(
        "out must not alias the operand; the kernel reads and writes overlapping memory");
  }
  if (detail::ranges_overlap<T>(scratch.data(), scratch.size(), operand.data(), operand.size()) ||
      detail::ranges_overlap<T>(scratch.data(), scratch.size(), out.data(), out.size())) {
    throw value_error("scratch must not overlap the operand or out");
  }
  detail::kron_execute<T>(plan, values.data(), ident.get(), operand.data(), out.data(),
                          scratch.data());
}

}  // namespace pantr::bspline
