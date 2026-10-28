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
#include "pantr/core/lazy.hpp"
#include "pantr/core/mdspan.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/core/simd.hpp"

#include "pantr/parallel/affinity.hpp"
#include "pantr/parallel/parallel_for.hpp"
#include "pantr/parallel/pool.hpp"
#include "pantr/parallel/work_model.hpp"
