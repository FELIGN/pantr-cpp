/// @file lagrange_variant.hpp
/// @brief The node families of the Lagrange basis, and their nodes on `[0, 1]`.
#pragma once

#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/quad/rules_1d.hpp"

#include <concepts>
#include <string>
#include <utility>

namespace pantr::basis {

/// Node family of a Lagrange basis. The values are the wire values the Python binding passes.
enum class LagrangeVariant : int {
  equispaces = 0,              ///< Equispaced, endpoints included (the trapezoidal nodes).
  gauss_legendre = 1,          ///< Gauss-Legendre nodes.
  gauss_lobatto_legendre = 2,  ///< Gauss-Lobatto-Legendre nodes (needs at least 2).
  chebyshev_1st = 3,           ///< Chebyshev-Gauss nodes of the first kind.
  chebyshev_2nd = 4,           ///< Chebyshev-Lobatto nodes, endpoints included (needs at least 2).
};

/// Convert a wire value to a `LagrangeVariant`.
///
/// Args:
///     value (int): The enumerator's value.
///
/// Returns:
///     LagrangeVariant: The variant.
///
/// Raises:
///     value_error: `value` names no variant.
[[nodiscard]] inline LagrangeVariant to_lagrange_variant(int value) {
  if (value < 0 || value > 4) {
    throw value_error("no Lagrange variant has the value " + std::to_string(value));
  }
  return static_cast<LagrangeVariant>(value);
}

/// Get the enumerator name Python's `LagrangeVariant` gives a variant (`EQUISPACES`, ...).
///
/// Args:
///     v (LagrangeVariant): The variant.
///
/// Returns:
///     const char*: The upper-case name.
[[nodiscard]] inline const char* variant_name(LagrangeVariant v) noexcept {
  switch (v) {
    case LagrangeVariant::equispaces:
      return "EQUISPACES";
    case LagrangeVariant::gauss_legendre:
      return "GAUSS_LEGENDRE";
    case LagrangeVariant::gauss_lobatto_legendre:
      return "GAUSS_LOBATTO_LEGENDRE";
    case LagrangeVariant::chebyshev_1st:
      return "CHEBYSHEV_1ST";
    case LagrangeVariant::chebyshev_2nd:
      return "CHEBYSHEV_2ND";
  }
  return "?";  // unreachable for a value built by to_lagrange_variant
}

/// Get the `n_pts` nodes of a Lagrange basis of the given family on `[0, 1]`.
///
/// Args:
///     v (LagrangeVariant): The node family.
///     n_pts (long long): Number of nodes; at least 1, or 2 for Gauss-Lobatto-Legendre and
///         Chebyshev-Lobatto.
///
/// Returns:
///     NdArray<T>: The nodes, ascending, shape `(n_pts,)`.
///
/// Raises:
///     value_error: `n_pts` is below the family's minimum.
template <std::floating_point T>
[[nodiscard]] NdArray<T> lagrange_points(LagrangeVariant v, long long n_pts) {
  switch (v) {
    case LagrangeVariant::equispaces:
      return std::move(quad::trapezoidal_1d<T>(n_pts).nodes);
    case LagrangeVariant::gauss_legendre:
      return std::move(quad::gauss_legendre_1d<T>(n_pts).nodes);
    case LagrangeVariant::gauss_lobatto_legendre:
      return std::move(quad::gauss_lobatto_legendre_1d<T>(n_pts).nodes);
    case LagrangeVariant::chebyshev_1st:
      return std::move(quad::chebyshev_gauss_1st_kind_1d<T>(n_pts).nodes);
    case LagrangeVariant::chebyshev_2nd:
      return quad::modified_chebyshev_nodes_1d<T>(n_pts);
  }
  throw value_error("unknown Lagrange variant");
}

}  // namespace pantr::basis
