// The windowing kernels of a distributed space (bspline/local_space.hpp, thb/local_space.hpp) and
// the coupling graph of a THB space. Oracles, each from a route other than the code under test:
//   - tensor product: each function's support read off the knot vector (a function is non-zero on
//     the interval `[u_j, u_{j+1}]` iff `knots[i] <= u_j` and `u_{j+1} <= knots[i + p + 1]`), never
//     the first-function-per-interval table the kernels use; the halo as "some function is
//     supported on both an owned cell and the cell" over every pair of cells and every function,
//     the owner as the first active cell of the support found by a scan over all cells in id order;
//   - THB: the same two definitions over `active_basis(cid)` of every pair of cells and every
//     cell, not the CSR incidence the kernels read;
//   - the THB coupling graph against the pairwise graph of the functions' supports, found from the
//     geometry (each active function's level support against every leaf cell, in the cells of the
//     deepest level), not from the space's per-cell table; and, on an unrefined space, against the
//     tensor-product graph of the root;
//   - the results at 1 and at up to 8 threads, which are exactly equal (each result entry is
//     computed alone, and the halo marks only the value 1).
#include <doctest/doctest.h>

#include "thb_test_support.hpp"

#include <pantr/bspline/coupling_graph.hpp>
#include <pantr/bspline/knot_factories.hpp>
#include <pantr/bspline/local_space.hpp>
#include <pantr/bspline/space.hpp>
#include <pantr/bspline/space_1d.hpp>
#include <pantr/core/errors.hpp>
#include <pantr/grid/hierarchical_grid.hpp>
#include <pantr/partition/coupling_graph.hpp>
#include <pantr/thb/coupling_graph.hpp>
#include <pantr/thb/local_space.hpp>
#include <pantr/thb/space.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace bs = pantr::bspline;
namespace th = pantr::thb;
namespace pp = pantr::partition;
using thb_test::check_many_threads_are_used;
using thb_test::eight_threads;
using thb_test::I64;
using thb_test::Ivec;
using thb_test::Lcg;
using thb_test::ThreadsGuard;
using Space = th::THBSplineSpace<double>;

// ---- Tensor-product setups -------------------------------------------------------------------

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

/// The setups of the brute-force comparisons: a degree (2, 3) space with a double interior knot
/// (a C^0 line, where a function's support is narrower), a degree (1, 2, 1) space of three
/// directions, a 1D cubic, a degree-0 space (every function on one interval) and a single Bezier
/// interval.
std::vector<TensorSetup> setups() {
  std::vector<TensorSetup> out;
  out.push_back(make_tensor({{0.0, 0.0, 0.0, 0.3, 0.55, 0.55, 0.8, 1.0, 1.0, 1.0},
                             {0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0}},
                            {2, 3}));
  out.push_back(make_tensor({{0.0, 0.0, 0.4, 0.7, 1.0, 1.0},
                             {0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0},
                             {0.0, 0.0, 0.2, 0.6, 0.9, 1.0, 1.0}},
                            {1, 2, 1}));
  out.push_back(make_tensor({{0.0, 0.0, 0.0, 0.0, 0.2, 0.5, 0.6, 0.9, 1.0, 1.0, 1.0, 1.0}}, {3}));
  out.push_back(make_tensor({{0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0}}, {0}));
  out.push_back(make_tensor({{0.0, 0.0, 0.0, 1.0, 1.0, 1.0}}, {2}));
  return out;
}

std::vector<std::size_t> shape_of(const TensorSetup& s, bool functions) {
  std::vector<std::size_t> shape;
  for (const auto& sp : s.space.spaces()) {
    shape.push_back(static_cast<std::size_t>(functions ? sp.num_basis() : sp.num_intervals()));
  }
  return shape;
}

std::size_t product(const std::vector<std::size_t>& shape) {
  std::size_t n = 1;
  for (const std::size_t e : shape) {
    n *= e;
  }
  return n;
}

std::vector<std::size_t> unflatten(std::size_t flat, const std::vector<std::size_t>& shape) {
  std::vector<std::size_t> multi(shape.size());
  for (std::size_t k = shape.size(); k-- > 0;) {
    multi[k] = flat % shape[k];
    flat /= shape[k];
  }
  return multi;
}

/// Per direction, per function, the intervals it is supported on, read off the knot vector.
using Supports = std::vector<std::vector<std::vector<char>>>;

Supports supports_from_knots(const TensorSetup& s) {
  Supports out;
  for (std::size_t k = 0; k < s.knots.size(); ++k) {
    const auto& knots = s.knots[k];
    const auto p = static_cast<std::size_t>(s.degrees[k]);
    const auto unique = s.space.spaces()[k].unique_knots_and_multiplicity(true).first;
    const std::size_t n_basis = knots.size() - p - 1;
    std::vector<std::vector<char>> per_function(n_basis, std::vector<char>(unique.size() - 1, 0));
    for (std::size_t i = 0; i < n_basis; ++i) {
      for (std::size_t j = 0; j + 1 < unique.size(); ++j) {
        per_function[i][j] =
            static_cast<char>(knots[i] <= unique[j] && unique[j + 1] <= knots[i + p + 1]);
      }
    }
    out.push_back(std::move(per_function));
  }
  return out;
}

/// Whether function `f` (multi-index) is supported on cell `c` (multi-index).
bool supported_on(const Supports& sup, const std::vector<std::size_t>& f,
                  const std::vector<std::size_t>& c) {
  for (std::size_t k = 0; k < f.size(); ++k) {
    if (sup[k][f[k]][c[k]] == 0) {
      return false;
    }
  }
  return true;
}

/// The halo by definition: a cell not owned that shares a function with an owned cell.
Ivec brute_halo(const TensorSetup& s, const Ivec& owned) {
  const Supports sup = supports_from_knots(s);
  const auto cells = shape_of(s, false);
  const auto funcs = shape_of(s, true);
  const std::size_t n_cells = product(cells);
  const std::size_t n_funcs = product(funcs);
  std::vector<char> is_owned(n_cells, 0);
  for (const I64 c : owned) {
    is_owned[static_cast<std::size_t>(c)] = 1;
  }
  Ivec halo;
  for (std::size_t other = 0; other < n_cells; ++other) {
    if (is_owned[other] != 0) {
      continue;
    }
    bool shares = false;
    for (std::size_t c = 0; c < n_cells && !shares; ++c) {
      for (std::size_t f = 0; f < n_funcs && is_owned[c] != 0 && !shares; ++f) {
        const auto fm = unflatten(f, funcs);
        shares = supported_on(sup, fm, unflatten(c, cells)) &&
                 supported_on(sup, fm, unflatten(other, cells));
      }
    }
    if (shares) {
      halo.push_back(static_cast<I64>(other));
    }
  }
  return halo;
}

/// The owner of every function by definition: the owner of the first active cell, in cell-id
/// order, whose cell the function is supported on.
std::vector<std::int32_t> brute_owner(const TensorSetup& s,
                                      const std::vector<std::int32_t>& owner) {
  const Supports sup = supports_from_knots(s);
  const auto cells = shape_of(s, false);
  const auto funcs = shape_of(s, true);
  std::vector<std::int32_t> out(product(funcs), -1);
  for (std::size_t f = 0; f < out.size(); ++f) {
    const auto fm = unflatten(f, funcs);
    for (std::size_t c = 0; c < owner.size(); ++c) {
      if (owner[c] >= 0 && supported_on(sup, fm, unflatten(c, cells))) {
        out[f] = owner[c];
        break;
      }
    }
  }
  return out;
}

/// `n` owners in `[0, ranks)`, each cell inactive (-1) with probability `inactive`.
std::vector<std::int32_t> random_owner(std::size_t n, std::int32_t ranks, double inactive,
                                       std::uint64_t seed) {
  Lcg rng(seed);
  std::vector<std::int32_t> out(n);
  for (auto& o : out) {
    o = rng.next() < inactive ? -1 : static_cast<std::int32_t>(rng.next() * ranks);
  }
  return out;
}

/// `count` cell ids in `[0, n)`, repeats possible.
Ivec random_ids(std::size_t n, std::size_t count, std::uint64_t seed) {
  Lcg rng(seed);
  Ivec out(count);
  for (auto& c : out) {
    c = static_cast<I64>(rng.next() * static_cast<double>(n));
  }
  return out;
}

// ---- Tensor product --------------------------------------------------------------------------

TEST_CASE("compute_halo: matches the cell-pair definition on every setup") {
  for (const TensorSetup& s : setups()) {
    const std::size_t n_cells = product(shape_of(s, false));
    std::vector<Ivec> owned_sets;
    owned_sets.push_back(Ivec{0});
    owned_sets.push_back(Ivec{static_cast<I64>(n_cells) - 1});
    owned_sets.push_back(random_ids(n_cells, 4, 1));
    owned_sets.push_back(random_ids(n_cells, 2 * n_cells, 2));  // repeats
    Ivec all(n_cells);
    for (std::size_t c = 0; c < n_cells; ++c) {
      all[c] = static_cast<I64>(c);
    }
    owned_sets.push_back(all);  // the halo of every cell is empty
    for (const Ivec& owned : owned_sets) {
      CHECK(bs::compute_halo(s.space, owned) == brute_halo(s, owned));
    }
    CHECK(bs::compute_halo(s.space, std::span<const I64>()).empty());
  }
}

TEST_CASE("compute_halo: the halo is sorted, owned cells are excluded") {
  const TensorSetup s = setups()[0];
  const Ivec owned = {7, 3, 3, 12};
  const Ivec halo = bs::compute_halo(s.space, owned);
  CHECK(std::is_sorted(halo.begin(), halo.end()));
  for (const I64 c : owned) {
    CHECK(std::find(halo.begin(), halo.end(), c) == halo.end());
  }
  CHECK(!halo.empty());
}

TEST_CASE("dof_owner: matches the first-active-cell definition on every setup") {
  for (const TensorSetup& s : setups()) {
    const std::size_t n_cells = product(shape_of(s, false));
    for (std::uint64_t seed = 1; seed <= 4; ++seed) {
      const auto owner = random_owner(n_cells, 3, 0.4, seed);
      CHECK(bs::dof_owner(s.space, owner) == brute_owner(s, owner));
    }
    // Every cell inactive: every function is dead.
    const std::vector<std::int32_t> none(n_cells, -1);
    const auto dead = bs::dof_owner(s.space, none);
    CHECK(std::all_of(dead.begin(), dead.end(), [](std::int32_t o) { return o == -1; }));
    CHECK(dead.size() == product(shape_of(s, true)));
  }
}

TEST_CASE("dof_owner: a function supported on one inactive cell is dead") {
  // Degree 1, open knots: the end functions live on the first and last interval only.
  const TensorSetup s = make_tensor({{0.0, 0.0, 0.5, 1.0, 1.0}}, {1});
  const std::vector<std::int32_t> owner = {-1, 4};
  const auto got = bs::dof_owner(s.space, owner);
  CHECK(got == std::vector<std::int32_t>{-1, 4, 4});
}

TEST_CASE("dof_owner_windowed: follows the request order and ignores repeats") {
  const TensorSetup s = setups()[0];
  const std::size_t n_cells = product(shape_of(s, false));
  const auto owner = random_owner(n_cells, 3, 0.3, 9);
  const auto all = bs::dof_owner(s.space, owner);
  const Ivec request = {13, 0, 13, 40, 5, static_cast<I64>(all.size()) - 1, 0};
  const auto got = bs::dof_owner_windowed(s.space, owner, request);
  REQUIRE(got.size() == request.size());
  for (std::size_t i = 0; i < request.size(); ++i) {
    CHECK(got[i] == all[static_cast<std::size_t>(request[i])]);
  }
  CHECK(bs::dof_owner_windowed(s.space, owner, std::span<const I64>()).empty());
}

TEST_CASE("local space kernels: every refusal and its message") {
  const TensorSetup s = setups()[0];
  const auto knots = bs::uniform_periodic_knots<double>(4, 2, 1, 0.0, 1.0);
  const bs::BsplineSpace<double> periodic(
      {s.space.spaces()[0], bs::BsplineSpace1D<double>(knots, 2, true)});
  const Ivec one = {0};
  const std::vector<std::int32_t> owner(1, 0);
  THB_THROWS_WITH_AS(bs::compute_halo(periodic, one), "periodic B-spline spaces are not supported.",
                     pantr::value_error);
  THB_THROWS_WITH_AS(bs::dof_owner(periodic, owner), "periodic B-spline spaces are not supported.",
                     pantr::value_error);
  THB_THROWS_WITH_AS(bs::dof_owner_windowed(periodic, owner, one),
                     "periodic B-spline spaces are not supported.", pantr::value_error);

  const auto n_cells = static_cast<I64>(product(shape_of(s, false)));
  const auto n_dofs = static_cast<I64>(product(shape_of(s, true)));
  const std::string cells = std::to_string(n_cells);
  const Ivec past = {0, n_cells};
  const Ivec negative = {-1};
  THB_THROWS_WITH_AS(bs::compute_halo(s.space, past),
                     ("owned cell id out of range [0, " + cells + ").").c_str(),
                     pantr::index_error);
  THB_THROWS_WITH_AS(bs::compute_halo(s.space, negative),
                     ("owned cell id out of range [0, " + cells + ").").c_str(),
                     pantr::index_error);

  const std::vector<std::int32_t> short_owner(static_cast<std::size_t>(n_cells) - 1, 0);
  const std::string want = "partition has " + std::to_string(n_cells - 1) + " cells; expected " +
                           cells + " (space.num_total_intervals).";
  THB_THROWS_WITH_AS(bs::dof_owner(s.space, short_owner), want.c_str(), pantr::value_error);
  THB_THROWS_WITH_AS(bs::dof_owner_windowed(s.space, short_owner, one), want.c_str(),
                     pantr::value_error);

  const std::vector<std::int32_t> good(static_cast<std::size_t>(n_cells), 0);
  const std::string dofs = "DOF id out of range [0, " + std::to_string(n_dofs) + ").";
  const Ivec bad_hi = {0, n_dofs};
  const Ivec bad_lo = {-1};
  THB_THROWS_WITH_AS(bs::dof_owner_windowed(s.space, good, bad_hi), dofs.c_str(),
                     pantr::index_error);
  THB_THROWS_WITH_AS(bs::dof_owner_windowed(s.space, good, bad_lo), dofs.c_str(),
                     pantr::index_error);
}

TEST_CASE("local space kernels: one thread and several give identical results") {
  check_many_threads_are_used();
  // Degree (2, 2) on 30 x 30 cells: 10^4 owned ids of the halo walk 25 cells each, and 4 x 10^4
  // requested functions walk up to 9, both far above the parallel threshold.
  const auto knots = thb_test::uniform_knots(30);
  const TensorSetup s = make_tensor({knots, knots}, {2, 2});
  const std::size_t n_cells = 900;
  const auto owner = random_owner(n_cells, 4, 0.5, 5);
  const Ivec owned = random_ids(n_cells, 10000, 6);
  const Ivec request = random_ids(32 * 32, 40000, 7);

  std::vector<Ivec> halos;
  std::vector<std::vector<std::int32_t>> owners;
  for (const std::size_t threads : {std::size_t{1}, eight_threads()}) {
    const ThreadsGuard guard(threads);
    halos.push_back(bs::compute_halo(s.space, owned));
    owners.push_back(bs::dof_owner_windowed(s.space, owner, request));
  }
  CHECK(halos[0] == halos[1]);
  CHECK(owners[0] == owners[1]);
  CHECK(halos[0] == brute_halo(s, [&] {
          Ivec unique = owned;
          std::sort(unique.begin(), unique.end());
          unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
          return unique;
        }()));
  const auto reference = brute_owner(s, owner);
  for (std::size_t i = 0; i < request.size(); i += 997) {
    CHECK(owners[1][i] == reference[static_cast<std::size_t>(request[i])]);
  }
}

// ---- THB -------------------------------------------------------------------------------------

/// The halo of a THB space by definition: a cell not owned sharing an active function (by
/// `active_basis`) with an owned cell.
Ivec brute_thb_halo(const Space& space, const Ivec& owned) {
  const I64 n = space.grid().num_cells();
  std::vector<char> is_owned(static_cast<std::size_t>(n), 0);
  for (const I64 c : owned) {
    is_owned[static_cast<std::size_t>(c)] = 1;
  }
  // The owned cells once each, in increasing id: the pairs are tested against these only.
  Ivec owned_list;
  for (I64 c = 0; c < n; ++c) {
    if (is_owned[static_cast<std::size_t>(c)] != 0) {
      owned_list.push_back(c);
    }
  }
  Ivec halo;
  for (I64 other = 0; other < n; ++other) {
    if (is_owned[static_cast<std::size_t>(other)] != 0) {
      continue;
    }
    const auto theirs = space.active_basis(other);
    bool shares = false;
    for (std::size_t at = 0; at < owned_list.size() && !shares; ++at) {
      for (const I64 d : space.active_basis(owned_list[at])) {
        shares = shares || std::find(theirs.begin(), theirs.end(), d) != theirs.end();
      }
    }
    if (shares) {
      halo.push_back(other);
    }
  }
  return halo;
}

/// The owner of every function of a THB space: the owner of the first active cell whose
/// `active_basis` lists it.
std::vector<std::int32_t> brute_thb_owner(const Space& space,
                                          const std::vector<std::int32_t>& owner) {
  std::vector<std::int32_t> out(static_cast<std::size_t>(space.num_total_basis()), -1);
  for (I64 dof = 0; dof < space.num_total_basis(); ++dof) {
    for (I64 c = 0; c < space.grid().num_cells(); ++c) {
      const auto basis = space.active_basis(c);
      if (owner[static_cast<std::size_t>(c)] >= 0 &&
          std::find(basis.begin(), basis.end(), dof) != basis.end()) {
        out[static_cast<std::size_t>(dof)] = owner[static_cast<std::size_t>(c)];
        break;
      }
    }
  }
  return out;
}

std::vector<Space> thb_spaces() {
  std::vector<Space> out;
  out.push_back(thb_test::refined_2d(true));
  out.push_back(thb_test::refined_2d(false));
  const auto base =
      thb_test::make_thb_space({thb_test::k2_3d, thb_test::k2_3d, thb_test::k2_3d}, {2, 2, 2},
                               {2, 2, 2}, {std::nullopt, std::nullopt, std::nullopt}, true);
  out.push_back(base.refine(Ivec{0}, std::nullopt));
  out.push_back(base);  // unrefined
  return out;
}

TEST_CASE("thb compute_halo: matches the shared-function definition") {
  for (const Space& space : thb_spaces()) {
    const auto n = static_cast<std::size_t>(space.grid().num_cells());
    std::vector<Ivec> owned_sets;
    owned_sets.push_back(Ivec{0});
    owned_sets.push_back(Ivec{static_cast<I64>(n) - 1});
    owned_sets.push_back(random_ids(n, 5, 3));
    owned_sets.push_back(random_ids(n, 3 * n, 4));  // repeats
    for (const Ivec& owned : owned_sets) {
      CHECK(th::compute_halo(space, owned) == brute_thb_halo(space, owned));
    }
    CHECK(th::compute_halo(space, std::span<const I64>()).empty());
  }
}

TEST_CASE("thb compute_halo: an id out of range raises as active_basis does") {
  const Space space = thb_test::refined_2d();
  const I64 n = space.grid().num_cells();
  const Ivec past = {0, n};
  const Ivec negative = {-3};
  const auto want = [&](I64 id) {
    return "cell id " + std::to_string(id) + " is out of range [0, " + std::to_string(n) + ").";
  };
  THB_THROWS_WITH_AS(th::compute_halo(space, past), want(n).c_str(), pantr::index_error);
  THB_THROWS_WITH_AS(th::compute_halo(space, negative), want(-3).c_str(), pantr::index_error);
}

TEST_CASE("thb dof_owner: matches the first-active-cell definition") {
  for (const Space& space : thb_spaces()) {
    const auto n = static_cast<std::size_t>(space.grid().num_cells());
    for (std::uint64_t seed = 1; seed <= 4; ++seed) {
      const auto owner = random_owner(n, 3, 0.4, seed);
      CHECK(th::dof_owner(space, owner) == brute_thb_owner(space, owner));
    }
    const std::vector<std::int32_t> none(n, -1);
    const auto dead = th::dof_owner(space, none);
    CHECK(dead.size() == static_cast<std::size_t>(space.num_total_basis()));
    CHECK(std::all_of(dead.begin(), dead.end(), [](std::int32_t o) { return o == -1; }));
  }
}

TEST_CASE("thb dof_owner: a function whose cells are all inactive is dead") {
  const Space space = thb_test::refined_2d();
  const auto n = static_cast<std::size_t>(space.grid().num_cells());
  // Only cell 0 is active: every function not listed on it is dead, the rest belong to rank 2.
  std::vector<std::int32_t> owner(n, -1);
  owner[0] = 2;
  const auto got = th::dof_owner(space, owner);
  const auto on_zero = space.active_basis(0);
  std::size_t alive = 0;
  for (std::size_t d = 0; d < got.size(); ++d) {
    const bool listed =
        std::find(on_zero.begin(), on_zero.end(), static_cast<I64>(d)) != on_zero.end();
    CHECK(got[d] == (listed ? 2 : -1));
    alive += listed ? 1 : 0;
  }
  CHECK(alive == on_zero.size());
  CHECK(alive < got.size());
}

TEST_CASE("thb dof_owner: the cell-count message") {
  const Space space = thb_test::refined_2d();
  const I64 n = space.grid().num_cells();
  const std::vector<std::int32_t> owner(static_cast<std::size_t>(n) + 2, 0);
  THB_THROWS_WITH_AS(th::dof_owner(space, owner),
                     ("partition has " + std::to_string(n + 2) + " cells; expected " +
                      std::to_string(n) + " (grid.num_cells).")
                         .c_str(),
                     pantr::value_error);
}

TEST_CASE("thb local space: one thread and several give identical results") {
  check_many_threads_are_used();
  // 130 x 130 root cells with a refined corner: about 1.7 x 10^4 cells, each holding at least the
  // 9 functions of its level, so the halo scan costs at least 1.5 x 10^5 operations, above the
  // 131072 below which it stays serial.
  const auto knots = thb_test::uniform_knots(130);
  const auto base =
      thb_test::make_thb_space({knots, knots}, {2, 2}, {2, 2}, {std::nullopt, std::nullopt}, true);
  const Space space = base.refine_region(0, Ivec{0, 0}, Ivec{6, 6}, std::nullopt);
  const auto n = static_cast<std::size_t>(space.grid().num_cells());
  const Ivec owned = random_ids(n, 300, 11);
  const auto owner = random_owner(n, 4, 0.3, 12);
  std::vector<Ivec> halos;
  std::vector<std::vector<std::int32_t>> owners;
  for (const std::size_t threads : {std::size_t{1}, eight_threads()}) {
    const ThreadsGuard guard(threads);
    halos.push_back(th::compute_halo(space, owned));
    owners.push_back(th::dof_owner(space, owner));
  }
  CHECK(halos[0] == halos[1]);
  CHECK(owners[0] == owners[1]);
  Ivec unique = owned;
  std::sort(unique.begin(), unique.end());
  unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
  CHECK(halos[0] == brute_thb_halo(space, unique));
}

// ---- THB coupling graph ----------------------------------------------------------------------

/// The functions whose support meets each cell, from the geometry alone. Every active function is
/// a level-`l` B-spline, listed by `active_function_indices(l)` (its global dof the level's offset
/// plus its position there) with per-direction support `[first_cell, last_cell]` of the level's 1D
/// space. Both it and a leaf cell are put in the cells of the deepest level, a level-`l` cell being
/// `factor^(deepest - l)` of them per direction, and the support meets the cell iff the two index
/// intervals overlap in every direction. Nothing here reads the per-cell table of the space
/// (`contributions`, `active_basis`, `cell_dof_incidence`).
std::vector<Ivec> brute_support_incidence(const Space& space) {
  const pantr::grid::HierarchicalGrid hier = space.grid();
  const std::size_t d = space.dim();
  const std::int64_t deepest = hier.max_level();
  const std::span<const I64> factor = hier.factor();
  // `scale[l][k] = factor[k]^(deepest - l)`: the deepest-level cells in one level-`l` cell.
  std::vector<Ivec> scale(static_cast<std::size_t>(deepest) + 1, Ivec(d, 1));
  for (std::int64_t l = deepest - 1; l >= 0; --l) {
    for (std::size_t k = 0; k < d; ++k) {
      scale[static_cast<std::size_t>(l)][k] = scale[static_cast<std::size_t>(l) + 1][k] * factor[k];
    }
  }
  const I64 n_cells = hier.num_cells();
  std::vector<Ivec> cell_lo(static_cast<std::size_t>(n_cells));
  std::vector<Ivec> cell_hi(static_cast<std::size_t>(n_cells));
  for (I64 c = 0; c < n_cells; ++c) {
    const auto cu = static_cast<std::size_t>(c);
    const Ivec midx = hier.cell_multi_index(c);
    const Ivec& unit = scale[static_cast<std::size_t>(hier.cell_level(c))];
    for (std::size_t k = 0; k < d; ++k) {
      cell_lo[cu].push_back(midx[k] * unit[k]);
      cell_hi[cu].push_back((midx[k] + 1) * unit[k]);
    }
  }
  std::vector<Ivec> cells_of(static_cast<std::size_t>(n_cells));
  for (std::int64_t l = 0; l < space.num_levels(); ++l) {
    const bs::BsplineSpace<double> level_space = space.level_space(l);
    const std::span<const I64> flat = space.active_function_indices(l);
    for (std::size_t at = 0; at < flat.size(); ++at) {
      // Unravel the row-major flat function index.
      Ivec lo(d);
      Ivec hi(d);
      I64 rest = flat[at];
      for (std::size_t k = d; k-- > 0;) {
        const I64 n_basis = level_space.spaces()[k].num_basis();
        const auto i = static_cast<std::size_t>(rest % n_basis);
        rest /= n_basis;
        const bs::detail::Support1D& sup = space.support(l, k);
        lo[k] = sup.first_cell[i] * scale[static_cast<std::size_t>(l)][k];
        hi[k] = (sup.last_cell[i] + 1) * scale[static_cast<std::size_t>(l)][k];
      }
      const I64 dof = space.func_offset()[static_cast<std::size_t>(l)] + static_cast<I64>(at);
      for (I64 c = 0; c < n_cells; ++c) {
        const auto cu = static_cast<std::size_t>(c);
        bool meets = true;
        for (std::size_t k = 0; k < d; ++k) {
          meets = meets && lo[k] < cell_hi[cu][k] && cell_lo[cu][k] < hi[k];
        }
        if (meets) {
          cells_of[cu].push_back(dof);
        }
      }
    }
  }
  return cells_of;
}

void check_same_graph(const pp::CouplingGraph& a, const pp::CouplingGraph& b) {
  CHECK(a.num_vertices() == b.num_vertices());
  CHECK(std::equal(a.xadj().begin(), a.xadj().end(), b.xadj().begin(), b.xadj().end()));
  CHECK(std::equal(a.adjncy().begin(), a.adjncy().end(), b.adjncy().begin(), b.adjncy().end()));
  CHECK(std::equal(a.edge_weights().begin(), a.edge_weights().end(), b.edge_weights().begin(),
                   b.edge_weights().end()));
  CHECK(std::equal(a.vertex_weights().begin(), a.vertex_weights().end(), b.vertex_weights().begin(),
                   b.vertex_weights().end()));
}

TEST_CASE("thb coupling_graph: equals the pairwise graph of the function supports") {
  // Oracle: two cells are adjacent with weight `w` iff exactly `w` active functions have a support
  // meeting both, found for every pair of cells from `brute_support_incidence` (geometry, not the
  // space's table). The incidence itself is compared with the space's table first, so a wrong
  // oracle would show there.
  for (const Space& space : thb_spaces()) {
    const std::vector<Ivec> cells_of = brute_support_incidence(space);
    const th::CellIncidence inc = space.cell_dof_incidence();
    const auto n = static_cast<std::size_t>(space.grid().num_cells());
    REQUIRE(cells_of.size() == n);
    for (std::size_t c = 0; c < n; ++c) {
      const std::span<const I64> table = std::span<const I64>(inc.dofs).subspan(
          static_cast<std::size_t>(inc.cell_ptr[c]),
          static_cast<std::size_t>(inc.cell_ptr[c + 1] - inc.cell_ptr[c]));
      CHECK(std::equal(table.begin(), table.end(), cells_of[c].begin(), cells_of[c].end()));
    }
    Ivec xadj(1, 0);
    Ivec adjncy;
    Ivec edge_weights;
    for (std::size_t a = 0; a < n; ++a) {
      for (std::size_t b = 0; b < n; ++b) {
        Ivec shared;
        std::set_intersection(cells_of[a].begin(), cells_of[a].end(), cells_of[b].begin(),
                              cells_of[b].end(), std::back_inserter(shared));
        if (a != b && !shared.empty()) {
          adjncy.push_back(static_cast<I64>(b));
          edge_weights.push_back(static_cast<I64>(shared.size()));
        }
      }
      xadj.push_back(static_cast<I64>(adjncy.size()));
    }
    const pp::CouplingGraph got = th::coupling_graph(space);
    CHECK(got.num_vertices() == static_cast<I64>(n));
    CHECK(std::equal(got.xadj().begin(), got.xadj().end(), xadj.begin(), xadj.end()));
    CHECK(std::equal(got.adjncy().begin(), got.adjncy().end(), adjncy.begin(), adjncy.end()));
    CHECK(std::equal(got.edge_weights().begin(), got.edge_weights().end(), edge_weights.begin(),
                     edge_weights.end()));
    CHECK(std::all_of(got.vertex_weights().begin(), got.vertex_weights().end(),
                      [](double w) { return w == 1.0; }));
    // With cell weights: the vertices carry them, the edges are unchanged.
    Lcg rng(21);
    std::vector<double> weights(n);
    for (double& w : weights) {
      w = 0.5 + rng.next();
    }
    const pp::CouplingGraph weighted = th::coupling_graph(space, std::span<const double>(weights));
    CHECK(std::equal(weighted.vertex_weights().begin(), weighted.vertex_weights().end(),
                     weights.begin(), weights.end()));
    CHECK(std::equal(weighted.xadj().begin(), weighted.xadj().end(), xadj.begin(), xadj.end()));
    CHECK(std::equal(weighted.adjncy().begin(), weighted.adjncy().end(), adjncy.begin(),
                     adjncy.end()));
    CHECK(std::equal(weighted.edge_weights().begin(), weighted.edge_weights().end(),
                     edge_weights.begin(), edge_weights.end()));
  }
}

TEST_CASE("thb coupling_graph: an unrefined space has the tensor-product graph of its root") {
  const auto knots = thb_test::uniform_knots(5);
  const Space space =
      thb_test::make_thb_space({knots, knots}, {2, 2}, {2, 2}, {std::nullopt, std::nullopt}, true);
  check_same_graph(th::coupling_graph(space), bs::coupling_graph(space.root_space()));
}

TEST_CASE("thb coupling_graph: invalid weights are refused") {
  const Space space = thb_test::refined_2d();
  const std::vector<double> short_weights = {1.0, 1.0};
  THB_THROWS_AS(th::coupling_graph(space, std::span<const double>(short_weights)),
                pantr::value_error);
  std::vector<double> bad(static_cast<std::size_t>(space.grid().num_cells()), 1.0);
  bad[1] = -1.0;
  THB_THROWS_WITH_AS(th::coupling_graph(space, std::span<const double>(bad)),
                     "cell_weights must be finite and non-negative.", pantr::value_error);
}

}  // namespace
