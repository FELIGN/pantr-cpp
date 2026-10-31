/// @file pantr.hpp
/// @brief Umbrella header: every public header of `pantr::pantr`.
///
/// Never includes an MPI header; the distributed layer (`pantr::mpi`) is included separately.
#pragma once

#include "pantr/config.hpp"
#include "pantr/version.hpp"

#include "pantr/core/errors.hpp"
#include "pantr/core/exact_rational.hpp"
#include "pantr/core/format.hpp"
#include "pantr/core/lambert_w.hpp"
#include "pantr/core/lazy.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/simd.hpp"

#include "pantr/parallel/affinity.hpp"
#include "pantr/parallel/parallel_for.hpp"
#include "pantr/parallel/pool.hpp"
#include "pantr/parallel/work_model.hpp"

#include "pantr/tolerance.hpp"

#include "pantr/quad/points_lattice.hpp"
#include "pantr/quad/quadrature_rule.hpp"
#include "pantr/quad/rules_1d.hpp"

#include "pantr/basis/bernstein.hpp"
#include "pantr/basis/cardinal_bspline.hpp"
#include "pantr/basis/lagrange.hpp"
#include "pantr/basis/lagrange_variant.hpp"
#include "pantr/basis/legendre.hpp"
#include "pantr/basis/tabulate.hpp"

#include "pantr/bspline/blossom.hpp"
#include "pantr/bspline/bspline.hpp"
#include "pantr/bspline/degree.hpp"
#include "pantr/bspline/derivative.hpp"
#include "pantr/bspline/derivative_rational.hpp"
#include "pantr/bspline/evaluate.hpp"
#include "pantr/bspline/extraction.hpp"
#include "pantr/bspline/knot_factories.hpp"
#include "pantr/bspline/knot_insertion.hpp"
#include "pantr/bspline/knot_removal.hpp"
#include "pantr/bspline/knots.hpp"
#include "pantr/bspline/periodic.hpp"
#include "pantr/bspline/product.hpp"
#include "pantr/bspline/reparametrize.hpp"
#include "pantr/bspline/restrict.hpp"
#include "pantr/bspline/roots.hpp"
#include "pantr/bspline/slice.hpp"
#include "pantr/bspline/space.hpp"
#include "pantr/bspline/space_1d.hpp"
#include "pantr/bspline/spanwise_extraction.hpp"
#include "pantr/bspline/to_beziers.hpp"
#include "pantr/change_basis.hpp"
