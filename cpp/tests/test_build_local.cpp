// `build_local` of a tensor-product and of a THB space (bspline/local_space.hpp,
// thb/local_space.hpp): the rank-local windowed space of a distributed space. Oracles, each from a
// route other than the code under test:
//   - the local basis equals the global basis pointwise: both spaces are evaluated at sample points
//     of every owned cell, the local values scattered into global function ids through
//     `local_to_global_dof`, and the two dense vectors compared (tolerance: the rounding model of
//     the Cox-de Boor recurrence and, for a THB space, of the truncation, `value_error_units`);
//   - the ownership masks over all ranks: each function that lies on an active cell is owned by
//     exactly one rank, and by the owner of the first active cell (in id order) it is supported on,
//     read off the knot vector (tensor) or `active_basis` (THB); a function on no active cell is
//     owned by none;
//   - the cell map: the bounding box of the window in the global multi-index, found here from the
//     cell ids by an independent unravel;
//   - every refusal, with its message and in its order.
#include <doctest/doctest.h>

#include "thb_test_support.hpp"

#include <pantr/bspline/knot_factories.hpp>
#include <pantr/bspline/local_space.hpp>
#include <pantr/bspline/space.hpp>
#include <pantr/bspline/space_1d.hpp>
#include <pantr/core/errors.hpp>
#include <pantr/core/mdspan.hpp>
#include <pantr/grid/partition.hpp>
#include <pantr/thb/local_space.hpp>
#include <pantr/thb/space.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace bs = pantr::bspline;
namespace th = pantr::thb;
using pantr::grid::Partition;
using thb_test::eps;
using thb_test::I64;
using thb_test::Ivec;
using thb_test::Lcg;
using Space = th::THBSplineSpace<double>;
using Owners = std::vector<std::int32_t>;

constexpr std::int64_t kRanks = 3;

/// Owner of cell `c` of `n`: `ranks` contiguous blocks of ids.
Owners block_owner(std::size_t n, std::int32_t ranks) {
  Owners out(n);
  for (std::size_t c = 0; c < n; ++c) {
    out[c] = static_cast<std::int32_t>((c * static_cast<std::size_t>(ranks)) / n);
  }
  return out;
}

/// `n` owners in `[0, ranks)`, each cell inactive (-1) with probability `inactive`.
Owners random_owner(std::size_t n, std::int32_t ranks, double inactive, std::uint64_t seed) {
  Lcg rng(seed);
  Owners out(n);
  for (auto& o : out) {
    o = rng.next() < inactive ? -1 : static_cast<std::int32_t>(rng.next() * ranks);
  }
  return out;
}

/// The partitions of a space of `n` cells the tests run over: contiguous blocks, a random
/// assignment with no inactive cell, and a random one with inactive cells. Each part owns a cell.
std::vector<Owners> partitions_of(std::size_t n) {
  std::vector<Owners> out;
  out.push_back(block_owner(n, static_cast<std::int32_t>(kRanks)));
  for (const auto& [inactive, seed] : {std::pair{0.0, 21ULL}, std::pair{0.3, 22ULL}}) {
    Owners o = random_owner(n, static_cast<std::int32_t>(kRanks), inactive, seed);
    // Cell `r` goes to rank `r`, so that no part is left without a cell by the random draw.
    REQUIRE(n >= static_cast<std::size_t>(kRanks));
    for (std::int32_t r = 0; r < kRanks; ++r) {
      o[static_cast<std::size_t>(r)] = r;
    }
    out.push_back(std::move(o));
  }
  return out;
}

// ---- Tensor product --------------------------------------------------------------------------

struct TensorSetup {
  std::vector<std::vector<double>> knots;
  std::vector<int> degrees;
  bs::BsplineSpace<double> space;
};

TensorSetup make_tensor(const std::vector<std::vector<double>>& knots,
                        const std::vector<int>& degrees) {
  std::vector<bs::BsplineSpace1D<double>> dirs;
  for (std::size_t k = 0; k < knots.size(); ++k) {
    dirs.emplace_back(std::span<const double>(knots[k]), degrees[k]);
  }
  return TensorSetup{knots, degrees, bs::BsplineSpace<double>(std::move(dirs))};
}

/// A degree (2, 3) space of 6 x 5 cells with a double interior knot, a degree (1, 2, 1) space, and
/// a 1D cubic: windows of one, two and three directions.
std::vector<TensorSetup> tensor_setups() {
  std::vector<TensorSetup> out;
  out.push_back(make_tensor({{0.0, 0.0, 0.0, 0.3, 0.55, 0.55, 0.8, 0.9, 1.0, 1.0, 1.0},
                             {0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 0.9, 1.0, 1.0, 1.0, 1.0}},
                            {2, 3}));
  out.push_back(make_tensor({{0.0, 0.0, 0.4, 0.7, 1.0, 1.0},
                             {0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0},
                             {0.0, 0.0, 0.2, 0.6, 0.9, 1.0, 1.0}},
                            {1, 2, 1}));
  out.push_back(make_tensor({{0.0, 0.0, 0.0, 0.0, 0.2, 0.5, 0.6, 0.9, 1.0, 1.0, 1.0, 1.0}}, {3}));
  return out;
}

I64 count_of(const TensorSetup& s, bool functions) {
  I64 n = 1;
  for (const auto& sp : s.space.spaces()) {
    n *= functions ? sp.num_basis() : sp.num_intervals();
  }
  return n;
}

/// The row-major unravel of `flat` over `shape`.
Ivec unravel(I64 flat, const Ivec& shape) {
  Ivec multi(shape.size());
  for (std::size_t k = shape.size(); k-- > 0;) {
    multi[k] = flat % shape[k];
    flat /= shape[k];
  }
  return multi;
}

I64 ravel(const Ivec& multi, const Ivec& shape) {
  I64 flat = 0;
  for (std::size_t k = 0; k < shape.size(); ++k) {
    flat = flat * shape[k] + multi[k];
  }
  return flat;
}

Ivec shape_of(const bs::BsplineSpace<double>& space, bool functions) {
  Ivec shape;
  for (const auto& sp : space.spaces()) {
    shape.push_back(functions ? sp.num_basis() : sp.num_intervals());
  }
  return shape;
}

/// Points of cell `cid` of a tensor space: `lo + t * (hi - lo)` for `t` in `ts` along every axis.
std::vector<double> tensor_cell_points(const bs::BsplineSpace<double>& space, I64 cid,
                                       const std::vector<double>& ts) {
  const std::size_t d = space.dim();
  const Ivec multi = unravel(cid, shape_of(space, false));
  std::vector<double> lo(d);
  std::vector<double> hi(d);
  for (std::size_t k = 0; k < d; ++k) {
    const auto u = space.spaces()[k].unique_knots_and_multiplicity(true).first;
    lo[k] = u[static_cast<std::size_t>(multi[k])];
    hi[k] = u[static_cast<std::size_t>(multi[k]) + 1];
  }
  std::size_t count = 1;
  for (std::size_t k = 0; k < d; ++k) {
    count *= ts.size();
  }
  std::vector<double> pts(count * d);
  for (std::size_t i = 0; i < count; ++i) {
    std::size_t rest = i;
    for (std::size_t k = d; k-- > 0;) {
      pts[i * d + k] = lo[k] + ts[rest % ts.size()] * (hi[k] - lo[k]);
      rest /= ts.size();
    }
  }
  return pts;
}

/// The values of `space` at the point `pt`, scattered into a vector over `n_global` global function
/// ids: local function `j` goes to `l2g[j]` (the identity when `l2g` is empty).
std::vector<double> dense_values(const bs::BsplineSpace<double>& space, const double* pt,
                                 const Ivec& l2g, I64 n_global) {
  const std::size_t d = space.dim();
  const auto [values, first] =
      space.tabulate_basis<double>(pantr::mdview<const double, 2>(pt, 1, d));
  const Ivec basis_shape = shape_of(space, true);
  Ivec orders;
  for (const auto& sp : space.spaces()) {
    orders.push_back(sp.degree() + 1);
  }
  std::vector<double> dense(static_cast<std::size_t>(n_global), 0.0);
  for (I64 j = 0; j < static_cast<I64>(space.num_local_functions()); ++j) {
    Ivec multi = unravel(j, orders);
    for (std::size_t k = 0; k < d; ++k) {
      multi[k] += first.data()[k];
    }
    const I64 local = ravel(multi, basis_shape);
    const I64 global = l2g.empty() ? local : l2g[static_cast<std::size_t>(local)];
    dense[static_cast<std::size_t>(global)] = values.data()[j];
  }
  return dense;
}

/// For every function (flat id), the intervals it is supported on, per direction, read off the
/// knot vector: function `i` is non-zero on `[u_j, u_{j+1}]` iff `knots[i] <= u_j` and
/// `u_{j+1} <= knots[i + p + 1]`.
std::vector<std::vector<std::vector<char>>> knot_supports(const TensorSetup& s) {
  std::vector<std::vector<std::vector<char>>> out;
  for (std::size_t k = 0; k < s.knots.size(); ++k) {
    const auto& knots = s.knots[k];
    const auto p = static_cast<std::size_t>(s.degrees[k]);
    const auto u = s.space.spaces()[k].unique_knots_and_multiplicity(true).first;
    const std::size_t n_basis = knots.size() - p - 1;
    std::vector<std::vector<char>> per_function(n_basis, std::vector<char>(u.size() - 1, 0));
    for (std::size_t i = 0; i < n_basis; ++i) {
      for (std::size_t j = 0; j + 1 < u.size(); ++j) {
        per_function[i][j] = static_cast<char>(knots[i] <= u[j] && u[j + 1] <= knots[i + p + 1]);
      }
    }
    out.push_back(std::move(per_function));
  }
  return out;
}

/// The owner of every function by definition: the owner of the first active cell, in id order,
/// that the function is supported on; -1 for none.
Owners tensor_owner_by_definition(const TensorSetup& s, const Owners& cell_owner) {
  const auto sup = knot_supports(s);
  const Ivec cells = shape_of(s.space, false);
  const Ivec funcs = shape_of(s.space, true);
  Owners out(static_cast<std::size_t>(count_of(s, true)), -1);
  for (std::size_t f = 0; f < out.size(); ++f) {
    const Ivec fm = unravel(static_cast<I64>(f), funcs);
    for (std::size_t c = 0; c < cell_owner.size() && out[f] < 0; ++c) {
      if (cell_owner[c] < 0) {
        continue;
      }
      const Ivec cm = unravel(static_cast<I64>(c), cells);
      bool on = true;
      for (std::size_t k = 0; k < fm.size(); ++k) {
        on = on && sup[k][static_cast<std::size_t>(fm[k])][static_cast<std::size_t>(cm[k])] != 0;
      }
      if (on) {
        out[f] = cell_owner[c];
      }
    }
  }
  return out;
}

TEST_CASE("build_local tensor: the local basis equals the global basis on the owned cells") {
  const std::vector<double> ts = {0.2, 0.5, 0.85};
  for (const TensorSetup& s : tensor_setups()) {
    const auto n_cells = static_cast<std::size_t>(count_of(s, false));
    const I64 n_dofs = count_of(s, true);
    for (const Owners& owner : partitions_of(n_cells)) {
      const Partition partition(owner, kRanks);
      for (std::int64_t rank = 0; rank < kRanks; ++rank) {
        const auto local = bs::build_local(s.space, partition, rank);
        REQUIRE(local.owned_cell_mask.size() == local.local_to_global_cell.size());
        const double tol =
            2.0 * thb_test::build_slack * eps *
            (2.0 * 3.0 * static_cast<double>(s.space.dim()) + static_cast<double>(s.space.dim()));
        std::size_t checked = 0;
        for (std::size_t l = 0; l < local.local_to_global_cell.size(); ++l) {
          if (local.owned_cell_mask[l] == 0) {
            continue;
          }
          const I64 g = local.local_to_global_cell[l];
          const auto pts = tensor_cell_points(s.space, g, ts);
          // The same points are inside local cell `l`: the cell's bounds are the global ones.
          const auto local_pts = tensor_cell_points(local.space, static_cast<I64>(l), ts);
          REQUIRE(pts.size() == local_pts.size());
          for (std::size_t i = 0; i < pts.size(); ++i) {
            CHECK(local_pts[i] == pts[i]);
          }
          const std::size_t d = s.space.dim();
          for (std::size_t i = 0; i < pts.size() / d; ++i) {
            const auto want = dense_values(s.space, pts.data() + i * d, {}, n_dofs);
            const auto got =
                dense_values(local.space, pts.data() + i * d, local.local_to_global_dof, n_dofs);
            for (std::size_t f = 0; f < want.size(); ++f) {
              CHECK(std::abs(got[f] - want[f]) <= tol);
            }
          }
          ++checked;
        }
        // The rank owns cells, and every one of them was compared.
        CHECK(checked == static_cast<std::size_t>(std::count(owner.begin(), owner.end(),
                                                             static_cast<std::int32_t>(rank))));
        CHECK(local.n_global_cells == static_cast<I64>(n_cells));
        CHECK(local.n_global_dofs == n_dofs);
      }
    }
  }
}

TEST_CASE("build_local tensor: the cell map is the bounding box of the window") {
  for (const TensorSetup& s : tensor_setups()) {
    const auto n_cells = static_cast<std::size_t>(count_of(s, false));
    const Ivec cell_shape = shape_of(s.space, false);
    for (const Owners& owner : partitions_of(n_cells)) {
      const Partition partition(owner, kRanks);
      for (std::int64_t rank = 0; rank < kRanks; ++rank) {
        const auto local = bs::build_local(s.space, partition, rank);
        const Ivec owned = partition.owned_cells(rank);
        const Ivec window_halo = bs::compute_halo(s.space, owned);
        // The bounding box of owned and halo cells, found from the multi-indices.
        Ivec lo = cell_shape;
        Ivec hi(cell_shape.size(), -1);
        for (const Ivec* ids : {&owned, &window_halo}) {
          for (const I64 c : *ids) {
            const Ivec m = unravel(c, cell_shape);
            for (std::size_t k = 0; k < m.size(); ++k) {
              lo[k] = (std::min)(lo[k], m[k]);
              hi[k] = (std::max)(hi[k], m[k]);
            }
          }
        }
        Ivec box_shape;
        for (std::size_t k = 0; k < lo.size(); ++k) {
          box_shape.push_back(hi[k] - lo[k] + 1);
        }
        I64 box_cells = 1;
        for (const I64 e : box_shape) {
          box_cells *= e;
        }
        REQUIRE(static_cast<I64>(local.local_to_global_cell.size()) == box_cells);
        // The local cells are the box in C order, and the local space has the box's shape.
        CHECK(shape_of(local.space, false) == box_shape);
        for (I64 l = 0; l < box_cells; ++l) {
          Ivec m = unravel(l, box_shape);
          for (std::size_t k = 0; k < m.size(); ++k) {
            m[k] += lo[k];
          }
          CHECK(local.local_to_global_cell[static_cast<std::size_t>(l)] == ravel(m, cell_shape));
        }
        // Owned cells are exactly the ones the partition gives to `rank`; owned and halo cells
        // are all inside the map.
        for (std::size_t l = 0; l < local.owned_cell_mask.size(); ++l) {
          const I64 g = local.local_to_global_cell[l];
          CHECK((local.owned_cell_mask[l] != 0) ==
                (owner[static_cast<std::size_t>(g)] == static_cast<std::int32_t>(rank)));
        }
        for (const Ivec* ids : {&owned, &window_halo}) {
          for (const I64 c : *ids) {
            CHECK(std::find(local.local_to_global_cell.begin(), local.local_to_global_cell.end(),
                            c) != local.local_to_global_cell.end());
          }
        }
        CHECK(static_cast<I64>(local.local_to_global_dof.size()) == [&] {
          I64 n = 1;
          for (const I64 e : shape_of(local.space, true)) {
            n *= e;
          }
          return n;
        }());
        CHECK(local.owned_dof_mask.size() == local.local_to_global_dof.size());
      }
    }
  }
}

TEST_CASE("build_local tensor: the dof masks over all ranks own each live function once") {
  for (const TensorSetup& s : tensor_setups()) {
    const auto n_cells = static_cast<std::size_t>(count_of(s, false));
    const auto n_dofs = static_cast<std::size_t>(count_of(s, true));
    for (const Owners& owner : partitions_of(n_cells)) {
      const Partition partition(owner, kRanks);
      const Owners expected = tensor_owner_by_definition(s, owner);
      std::vector<int> times(n_dofs, 0);
      Owners owner_of(n_dofs, -1);
      for (std::int64_t rank = 0; rank < kRanks; ++rank) {
        const auto local = bs::build_local(s.space, partition, rank);
        for (std::size_t j = 0; j < local.owned_dof_mask.size(); ++j) {
          if (local.owned_dof_mask[j] != 0) {
            const auto g = static_cast<std::size_t>(local.local_to_global_dof[j]);
            ++times[g];
            owner_of[g] = static_cast<std::int32_t>(rank);
          }
        }
      }
      for (std::size_t g = 0; g < n_dofs; ++g) {
        CHECK(times[g] == (expected[g] >= 0 ? 1 : 0));
        CHECK(owner_of[g] == expected[g]);
      }
    }
  }
}

TEST_CASE("build_local tensor: a single part owns every cell and function") {
  const TensorSetup s = tensor_setups()[0];
  const auto n_cells = static_cast<std::size_t>(count_of(s, false));
  const Partition partition(Owners(n_cells, 0), 1);
  const auto local = bs::build_local(s.space, partition, 0);
  CHECK(shape_of(local.space, false) == shape_of(s.space, false));
  CHECK(std::all_of(local.owned_cell_mask.begin(), local.owned_cell_mask.end(),
                    [](std::uint8_t m) { return m == 1; }));
  CHECK(std::all_of(local.owned_dof_mask.begin(), local.owned_dof_mask.end(),
                    [](std::uint8_t m) { return m == 1; }));
  for (std::size_t i = 0; i < local.local_to_global_cell.size(); ++i) {
    CHECK(local.local_to_global_cell[i] == static_cast<I64>(i));
  }
  for (std::size_t i = 0; i < local.local_to_global_dof.size(); ++i) {
    CHECK(local.local_to_global_dof[i] == static_cast<I64>(i));
  }
}

TEST_CASE("build_local tensor: every refusal and its message, in order") {
  const TensorSetup s = tensor_setups()[0];
  const auto n_cells = static_cast<std::size_t>(count_of(s, false));
  const Partition partition(block_owner(n_cells, 3), 3);

  // Periodic first, whatever else is wrong.
  const auto knots = bs::uniform_periodic_knots<double>(4, 2, 1, 0.0, 1.0);
  const bs::BsplineSpace<double> periodic(
      {s.space.spaces()[0], bs::BsplineSpace1D<double>(knots, 2, true)});
  THB_THROWS_WITH_AS(bs::build_local(periodic, partition, 7),
                     "periodic B-spline spaces are not supported.", pantr::value_error);

  THB_THROWS_WITH_AS(bs::build_local(s.space, partition, 3), "rank must be in [0, 3); got 3.",
                     pantr::value_error);
  THB_THROWS_WITH_AS(bs::build_local(s.space, partition, -1), "rank must be in [0, 3); got -1.",
                     pantr::value_error);

  // A rank out of range is reported before a cell-count mismatch.
  const Partition wrong_count(Owners(n_cells + 2, 0), 3);
  THB_THROWS_WITH_AS(bs::build_local(s.space, wrong_count, 5), "rank must be in [0, 3); got 5.",
                     pantr::value_error);
  THB_THROWS_WITH_AS(bs::build_local(s.space, wrong_count, 0),
                     ("partition has " + std::to_string(n_cells + 2) + " cells; expected " +
                      std::to_string(n_cells) + " (space.num_total_intervals).")
                         .c_str(),
                     pantr::value_error);

  // A mismatch is reported before a rank owning nothing: `wrong_count` gives rank 1 no cell.
  THB_THROWS_WITH_AS(bs::build_local(s.space, wrong_count, 1),
                     ("partition has " + std::to_string(n_cells + 2) + " cells; expected " +
                      std::to_string(n_cells) + " (space.num_total_intervals).")
                         .c_str(),
                     pantr::value_error);

  const Partition empty_rank(Owners(n_cells, 0), 3);
  THB_THROWS_WITH_AS(bs::build_local(s.space, empty_rank, 2),
                     "rank 2 owns no cells; cannot build a local space.", pantr::value_error);
  // Only inactive cells besides: still no cell for the rank.
  Owners some_inactive(n_cells, -1);
  some_inactive[0] = 0;
  const Partition mostly_inactive(some_inactive, 2);
  THB_THROWS_WITH_AS(bs::build_local(s.space, mostly_inactive, 1),
                     "rank 1 owns no cells; cannot build a local space.", pantr::value_error);
  THB_NOTHROW(bs::build_local(s.space, mostly_inactive, 0));
}

// ---- THB -------------------------------------------------------------------------------------

std::vector<Space> thb_spaces() {
  std::vector<Space> out;
  out.push_back(thb_test::refined_2d(true));
  out.push_back(thb_test::refined_2d(false));
  const auto base =
      thb_test::make_thb_space({thb_test::k2_3d, thb_test::k2_3d, thb_test::k2_3d}, {2, 2, 2},
                               {2, 2, 2}, {std::nullopt, std::nullopt, std::nullopt}, true);
  out.push_back(base.refine(Ivec{0}, std::nullopt));
  return out;
}

/// The owner of every function by definition: the owner of the first active cell whose
/// `active_basis` lists it; -1 for none.
Owners thb_owner_by_definition(const Space& space, const Owners& cell_owner) {
  Owners out(static_cast<std::size_t>(space.num_total_basis()), -1);
  for (I64 c = 0; c < space.grid().num_cells(); ++c) {
    if (cell_owner[static_cast<std::size_t>(c)] < 0) {
      continue;
    }
    for (const I64 d : space.active_basis(c)) {
      auto& slot = out[static_cast<std::size_t>(d)];
      if (slot < 0) {
        slot = cell_owner[static_cast<std::size_t>(c)];
      }
    }
  }
  return out;
}

/// A 10 x 10 root cells space refined twice over interior boxes: windows of a few cells fall
/// inside the domain, where the function selection of a restricted space can differ from the
/// global one at the window boundary.
Space interior_refined_2d() {
  const auto knots = thb_test::uniform_knots(10);
  const auto base =
      thb_test::make_thb_space({knots, knots}, {2, 2}, {2, 2}, {std::nullopt, std::nullopt}, true);
  const auto once = base.refine_region(0, Ivec{2, 2}, Ivec{7, 7}, std::nullopt);
  return once.refine_region(1, Ivec{6, 6}, Ivec{12, 12}, std::nullopt);
}

/// Check, for each rank of `partition`, that the local basis equals the global one at sample
/// points of every owned cell: same function ids (through `local_to_global_dof`), values equal
/// within the rounding model.
void check_thb_basis(const Space& space, const Owners& owner, std::int64_t n_parts) {
  const std::vector<double> ts = {0.2, 0.5, 0.85};
  const Partition partition(owner, n_parts);
  const auto n_cells = static_cast<std::size_t>(space.grid().num_cells());
  for (std::int64_t rank = 0; rank < n_parts; ++rank) {
    const auto local = th::build_local(space, partition, rank);
    REQUIRE(local.owned_cell_mask.size() == local.local_to_global_cell.size());
    REQUIRE(static_cast<I64>(local.local_to_global_cell.size()) == local.space.grid().num_cells());
    REQUIRE(static_cast<I64>(local.local_to_global_dof.size()) == local.space.num_total_basis());
    const double tol =
        thb_test::build_slack * eps *
        (thb_test::value_error_units(space) + thb_test::value_error_units(local.space));
    std::size_t checked = 0;
    for (std::size_t l = 0; l < local.local_to_global_cell.size(); ++l) {
      if (local.owned_cell_mask[l] == 0) {
        continue;
      }
      const I64 g = local.local_to_global_cell[l];
      const std::vector<double> pts = thb_test::cell_points(space, g, ts);
      const std::size_t d = space.dim();
      const pantr::mdview<const double, 2> rows(pts.data(), pts.size() / d, d);
      const th::TabulatedBasis want = space.tabulate_basis(g, rows);
      const th::TabulatedBasis got = local.space.tabulate_basis(static_cast<I64>(l), rows);
      REQUIRE(want.dofs.size() == got.dofs.size());
      // The local function ids mapped to global ones, in the global order.
      std::vector<std::pair<I64, std::size_t>> mapped;
      for (std::size_t j = 0; j < got.dofs.size(); ++j) {
        const I64 global = local.local_to_global_dof[static_cast<std::size_t>(got.dofs.data()[j])];
        REQUIRE(global >= 0);
        mapped.emplace_back(global, j);
      }
      std::sort(mapped.begin(), mapped.end());
      const std::size_t n_active = want.dofs.size();
      for (std::size_t j = 0; j < n_active; ++j) {
        CHECK(mapped[j].first == want.dofs.data()[j]);
        for (std::size_t i = 0; i < pts.size() / d; ++i) {
          CHECK(std::abs(got.values.data()[i * n_active + mapped[j].second] -
                         want.values.data()[i * n_active + j]) <= tol);
        }
      }
      ++checked;
    }
    CHECK(checked == static_cast<std::size_t>(
                         std::count(owner.begin(), owner.end(), static_cast<std::int32_t>(rank))));
    CHECK(local.n_global_cells == static_cast<I64>(n_cells));
    CHECK(local.n_global_dofs == space.num_total_basis());
  }
}

/// Check the cell map and both masks against `owner`, and that over all ranks each function on an
/// active cell is owned once, by the owner given by the definition.
void check_thb_maps_and_masks(const Space& space, const Owners& owner, std::int64_t n_parts) {
  const Partition partition(owner, n_parts);
  const auto n_cells = static_cast<std::size_t>(space.grid().num_cells());
  const auto n_dofs = static_cast<std::size_t>(space.num_total_basis());
  const Owners expected = thb_owner_by_definition(space, owner);
  std::vector<int> times(n_dofs, 0);
  Owners owner_of(n_dofs, -1);
  for (std::int64_t rank = 0; rank < n_parts; ++rank) {
    const auto local = th::build_local(space, partition, rank);
    // Each owned and halo cell is in the map; the map holds distinct valid global ids.
    const Ivec owned = partition.owned_cells(rank);
    const Ivec halo = th::compute_halo(space, owned);
    const std::set<I64> mapped(local.local_to_global_cell.begin(),
                               local.local_to_global_cell.end());
    CHECK(mapped.size() == local.local_to_global_cell.size());
    for (const Ivec* ids : {&owned, &halo}) {
      for (const I64 c : *ids) {
        CHECK(mapped.count(c) == 1);
      }
    }
    for (std::size_t l = 0; l < local.owned_cell_mask.size(); ++l) {
      const I64 g = local.local_to_global_cell[l];
      REQUIRE(g >= 0);
      REQUIRE(g < static_cast<I64>(n_cells));
      CHECK((local.owned_cell_mask[l] != 0) ==
            (owner[static_cast<std::size_t>(g)] == static_cast<std::int32_t>(rank)));
    }
    for (std::size_t j = 0; j < local.owned_dof_mask.size(); ++j) {
      const I64 g = local.local_to_global_dof[j];
      if (g < 0) {
        // A function with no global counterpart is never owned.
        CHECK(local.owned_dof_mask[j] == 0);
      } else if (local.owned_dof_mask[j] != 0) {
        ++times[static_cast<std::size_t>(g)];
        owner_of[static_cast<std::size_t>(g)] = static_cast<std::int32_t>(rank);
      }
    }
  }
  for (std::size_t g = 0; g < n_dofs; ++g) {
    CHECK(times[g] == (expected[g] >= 0 ? 1 : 0));
    CHECK(owner_of[g] == expected[g]);
  }
}

TEST_CASE("build_local thb: the local basis equals the global basis on the owned cells") {
  std::vector<Space> spaces = thb_spaces();
  spaces.push_back(interior_refined_2d());
  for (const Space& space : spaces) {
    for (const Owners& owner : partitions_of(static_cast<std::size_t>(space.grid().num_cells()))) {
      check_thb_basis(space, owner, kRanks);
    }
  }
}

TEST_CASE("build_local thb: the maps and masks agree with the partition") {
  std::vector<Space> spaces = thb_spaces();
  spaces.push_back(interior_refined_2d());
  for (const Space& space : spaces) {
    for (const Owners& owner : partitions_of(static_cast<std::size_t>(space.grid().num_cells()))) {
      check_thb_maps_and_masks(space, owner, kRanks);
    }
  }
}

TEST_CASE("build_local thb: a function with no global counterpart is in the map and not owned") {
  // Rank 1 owns cell 0 and one interior cell, rank 0 the rest: the interior cell's window ends
  // inside the domain, where the restricted space selects functions the global one does not have.
  const Space space = interior_refined_2d();
  const auto n_cells = static_cast<std::size_t>(space.grid().num_cells());
  std::size_t windows_with_gap = 0;
  for (const std::size_t cell : {std::size_t{4}, std::size_t{14}, std::size_t{36}}) {
    REQUIRE(cell < n_cells);
    Owners owner(n_cells, 0);
    owner[cell] = 1;
    // Cell 0 as well, so that rank 1 owns function 0: a -1 read as an index would mark it owned.
    owner[0] = 1;
    const Partition partition(owner, 2);
    const auto local = th::build_local(space, partition, 1);
    const auto gaps =
        std::count(local.local_to_global_dof.begin(), local.local_to_global_dof.end(), I64{-1});
    windows_with_gap += gaps > 0 ? 1 : 0;
    for (std::size_t j = 0; j < local.owned_dof_mask.size(); ++j) {
      if (local.local_to_global_dof[j] < 0) {
        CHECK(local.owned_dof_mask[j] == 0);
      }
    }
    check_thb_basis(space, owner, 2);
    check_thb_maps_and_masks(space, owner, 2);
  }
  // The cases were chosen to contain such a function; if the selection changes and none does, the
  // test no longer exercises the -1 entries.
  CHECK(windows_with_gap == 3);
}

TEST_CASE("build_local thb: every refusal and its message, in order") {
  const Space space = thb_test::refined_2d();
  const I64 n = space.grid().num_cells();
  const auto un = static_cast<std::size_t>(n);
  const Partition partition(block_owner(un, 3), 3);

  THB_THROWS_WITH_AS(th::build_local(space, partition, 3), "rank must be in [0, 3); got 3.",
                     pantr::value_error);
  THB_THROWS_WITH_AS(th::build_local(space, partition, -1), "rank must be in [0, 3); got -1.",
                     pantr::value_error);

  const Partition wrong_count(Owners(un + 2, 0), 3);
  const std::string count_message = "partition has " + std::to_string(n + 2) + " cells; expected " +
                                    std::to_string(n) + " (grid.num_cells).";
  THB_THROWS_WITH_AS(th::build_local(space, wrong_count, 5), "rank must be in [0, 3); got 5.",
                     pantr::value_error);
  THB_THROWS_WITH_AS(th::build_local(space, wrong_count, 0), count_message.c_str(),
                     pantr::value_error);
  // A mismatch is reported before a rank owning nothing.
  THB_THROWS_WITH_AS(th::build_local(space, wrong_count, 1), count_message.c_str(),
                     pantr::value_error);

  const Partition empty_rank(Owners(un, 0), 3);
  THB_THROWS_WITH_AS(th::build_local(space, empty_rank, 2),
                     "rank 2 owns no cells; cannot build a local space.", pantr::value_error);
  Owners some_inactive(un, -1);
  some_inactive[0] = 0;
  const Partition mostly_inactive(some_inactive, 2);
  THB_THROWS_WITH_AS(th::build_local(space, mostly_inactive, 1),
                     "rank 1 owns no cells; cannot build a local space.", pantr::value_error);
  THB_NOTHROW(th::build_local(space, mostly_inactive, 0));
}

}  // namespace
