/// @file kron_apply.hpp
/// @brief Applying a Kronecker product `M = kron(M_0, ..., M_{d-1})` to a vector or on both sides
/// of
///        a matrix, as one mode product per direction (and per side).
///
/// `M v` is `d` mode products on `v` viewed as a tensor with the extents `(n_in_0, ...,
/// n_in_{d-1})`, and `M^T v` the same with the transposed factors. `M^T K M` and `M K M^T` are `2d`
/// mode products on the matrix `K` viewed as a tensor with the extents of its rows followed by
/// those of its columns, the row mode and the column mode of a direction being contracted with the
/// same factor, one after the other. A direction whose factor is the identity is skipped. This one
/// primitive replaces the specialised kernels the Python implementation had for each combination
/// of operation and dimension.
///
/// The geometry of the products depends only on the factors' shapes, so it is worked out once as a
/// `KronPlan` and shared by every cell of a batch; the identity pattern and the factors' values
/// are per cell.
#pragma once

#include "pantr/bezier/detail/along_axis.hpp"
#include "pantr/bspline/detail/mode_product.hpp"
#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/scalar.hpp"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace pantr::bspline {

/// Which product of the element operator `M = kron(M_0, ..., M_{d-1})` an application computes.
///
/// The values are the wire values the Python binding passes.
enum class ExtractionOp : int {
  apply = 0,            ///< `M v`: `v` has `prod(n_in)` entries, the result `prod(n_out)`.
  apply_transpose = 1,  ///< `M^T v`: `v` has `prod(n_out)` entries, the result `prod(n_in)`.
  mt_k_m = 2,           ///< `M^T K M`: `K` is `prod(n_out)` square, the result `prod(n_in)` square.
  m_k_mt = 3,           ///< `M K M^T`: `K` is `prod(n_in)` square, the result `prod(n_out)` square.
};

/// Convert a wire value to an `ExtractionOp`.
///
/// Raises:
///     value_error: `value` names no operation.
[[nodiscard]] inline ExtractionOp to_extraction_op(int value) {
  if (value < 0 || value > 3) {
    throw value_error("no extraction operation has the value " + std::to_string(value));
  }
  return static_cast<ExtractionOp>(value);
}

/// One direction's factor of a Kronecker product.
///
/// Attributes:
///     values (std::span<const T>): `rows * cols` values, row-major. Not read when `identity`.
///     rows (std::size_t): `n_out`: rows of the factor.
///     cols (std::size_t): `n_in`: columns of the factor.
///     identity (bool): The factor is the identity (it must be square), so its direction is
///         skipped.
template <StorageFloat T>
struct KronFactor {
  std::span<const T> values;
  std::size_t rows;
  std::size_t cols;
  bool identity;
};

namespace detail {

/// One mode product of a plan: the direction whose factor it applies and the geometry.
///
/// Attributes:
///     direction (std::size_t): The factor, an index into the per-direction arrays.
///     shape (ModeShape): The geometry of the raw mode product.
struct KronStage {
  std::size_t direction;
  ModeShape shape;
};

/// The stages of one operation on factors of given shapes, in execution order.
///
/// Attributes:
///     stages (std::vector<KronStage>): `d` stages, or `2d` (row mode then column mode, per
///         direction) for the two-sided operations.
///     operand_size (std::size_t): Entries of the operand (a vector, or a square matrix).
///     result_size (std::size_t): Entries of the result.
///     operand_side (std::size_t): Entries of the operand's vector, or the side of its square
///         matrix: the product of the extents the operation reads.
///     result_side (std::size_t): The same for the result.
///     half (std::size_t): Entries of the larger intermediate result, so of each of the two
///         buffers the stages alternate between (0 when the one stage writes the result).
///     ops (double): Multiply-adds, doubled, of all stages: the work of one cell with no identity
///         direction, at least 1.
struct KronPlan {
  std::vector<KronStage> stages;
  std::size_t operand_size = 0;
  std::size_t result_size = 0;
  std::size_t operand_side = 0;
  std::size_t result_side = 0;
  std::size_t half = 0;
  double ops = 1.0;
};

/// Work out the plan of an operation.
///
/// Args:
///     op (ExtractionOp): The operation.
///     rows (std::span<const std::size_t>): `n_out` per direction, at least one direction.
///     cols (std::span<const std::size_t>): `n_in` per direction, as many as `rows`.
///
/// Returns:
///     KronPlan: The plan.
///
/// Raises:
///     value_error: `rows` and `cols` differ in length, or are empty, or an extent is 0.
///     capacity_error: A product of extents overflows `std::size_t`.
[[nodiscard]] inline KronPlan make_kron_plan(ExtractionOp op, std::span<const std::size_t> rows,
                                             std::span<const std::size_t> cols) {
  const std::size_t d = rows.size();
  if (cols.size() != d) {
    throw value_error(
        "input_shape_per_dir and output_shape_per_dir must have the same length; "
        "got " +
        std::to_string(cols.size()) + " and " + std::to_string(rows.size()));
  }
  if (d == 0) {
    throw value_error("At least one direction is required");
  }
  for (std::size_t k = 0; k < d; ++k) {
    if (rows[k] == 0 || cols[k] == 0) {
      throw value_error("a factor of the Kronecker product has no rows or no columns.");
    }
  }
  // The operation reads the `from` extents and writes the `to` ones; a transposed operation reads
  // the rows of the stored factors.
  const bool transposed = op == ExtractionOp::apply_transpose || op == ExtractionOp::mt_k_m;
  const bool two_sided = op == ExtractionOp::mt_k_m || op == ExtractionOp::m_k_mt;
  const std::span<const std::size_t> from = transposed ? rows : cols;
  const std::span<const std::size_t> to = transposed ? cols : rows;
  const std::size_t sides = two_sided ? 2 : 1;
  std::vector<std::size_t> extents(sides * d);
  for (std::size_t k = 0; k < d; ++k) {
    extents[k] = from[k];
    if (two_sided) {
      extents[d + k] = from[k];
    }
  }
  KronPlan plan;
  plan.operand_size = checked_product(extents);
  plan.operand_side = checked_product(std::span<const std::size_t>(extents).first(d));
  double ops = 0.0;
  for (std::size_t k = 0; k < d; ++k) {
    for (std::size_t side = 0; side < sides; ++side) {
      const std::size_t mode = side * d + k;
      const bezier::detail::AxisSplit around = bezier::detail::split_around(extents, mode);
      plan.stages.push_back(KronStage{.direction = k,
                                      .shape = ModeShape{.ld = cols[k],
                                                         .n_out = to[k],
                                                         .n_in = from[k],
                                                         .x_extent = from[k],
                                                         .y_extent = to[k],
                                                         .outer = around.outer,
                                                         .inner = around.inner,
                                                         .transpose = transposed}});
      extents[mode] = to[k];
      ops += 2.0 * static_cast<double>(around.outer) * static_cast<double>(to[k]) *
             static_cast<double>(from[k]) * static_cast<double>(around.inner);
      const std::size_t after = checked_product(extents);
      const bool last = k + 1 == d && side + 1 == sides;
      if (!last) {
        plan.half = (std::max)(plan.half, after);
      }
    }
  }
  plan.result_size = checked_product(extents);
  plan.result_side = checked_product(std::span<const std::size_t>(extents).first(d));
  plan.ops = (std::max)(ops, 1.0);
  return plan;
}

/// Run a plan on one cell.
///
/// The stages of the non-identity directions are applied in order: the first reads `x`, the last
/// writes `y`, and those between them alternate between the two halves of `scratch`. With every
/// direction identity the cell is a copy of `x` into `y`.
///
/// Args:
///     plan (const KronPlan&): The plan.
///     values (const T* const*): Per direction, the first value of the factor (read only where the
///         direction is not identity).
///     identity (const bool*): Per direction, whether the factor is the identity.
///     x (const T*): The operand, `plan.operand_size` entries.
///     y (T*): The result, `plan.result_size` entries.
///     scratch (T*): `2 * plan.half` entries, when a stage but the last one runs.
///
/// Note:
///     Inputs are assumed to be correct (no validation performed). `x` and `y` must not overlap
///     unless every direction is identity; `scratch` overlaps neither. `noexcept`, and calls the
///     contracting kernel only, so the sums contract into fused multiply-adds.
template <StorageFloat T>
void kron_execute(const KronPlan& plan, const T* const* values, const bool* identity, const T* x,
                  T* y, T* scratch) noexcept {
  const std::size_t n_stages = plan.stages.size();
  std::size_t last = n_stages;  // one past the last stage that runs
  for (std::size_t s = n_stages; s-- > 0;) {
    if (!identity[plan.stages[s].direction]) {
      last = s;
      break;
    }
  }
  if (last == n_stages) {
    // Every direction is identity: operand and result have the same size.
    std::memmove(y, x, plan.operand_size * sizeof(T));
    return;
  }
  const T* source = x;
  std::size_t flip = 0;
  for (std::size_t s = 0; s <= last; ++s) {
    const KronStage& stage = plan.stages[s];
    if (identity[stage.direction]) {
      continue;
    }
    T* target = s == last ? y : scratch + flip * plan.half;
    mode_product_kernel<T>(values[stage.direction], source, target, stage.shape);
    source = target;
    flip ^= 1;
  }
}

/// Whether two ranges of memory overlap (adjacent ranges do not).
template <class T>
[[nodiscard]] inline bool ranges_overlap(const T* a, std::size_t na, const T* b,
                                         std::size_t nb) noexcept {
  const std::less<const T*> before;
  return before(a, b + nb) && before(b, a + na) && na != 0 && nb != 0;
}

}  // namespace detail

/// Get the scratch an operation needs to run on a cell, in entries: twice the larger intermediate
/// result of its stages, or 0 when one stage does everything.
///
/// Sufficient for any identity pattern, since an identity direction keeps its extent.
///
/// Args:
///     op (ExtractionOp): The operation.
///     rows (std::span<const std::size_t>): `n_out` per direction.
///     cols (std::span<const std::size_t>): `n_in` per direction.
///
/// Returns:
///     std::size_t: The number of entries.
///
/// Raises:
///     value_error: As `detail::make_kron_plan`.
///     capacity_error: A product of extents overflows `std::size_t`.
[[nodiscard]] inline std::size_t required_scratch_size(ExtractionOp op,
                                                       std::span<const std::size_t> rows,
                                                       std::span<const std::size_t> cols) {
  return 2 * detail::make_kron_plan(op, rows, cols).half;
}

}  // namespace pantr::bspline
