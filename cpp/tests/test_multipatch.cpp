// Interfaces between patches: `Interface`, `match_face_cps`, `detect_interfaces`,
// `verify_interface`, `MultiPatch`, and the storage-type conversion `bspline::astype` the binding
// uses for patches of mixed dtypes.
//
// Oracles. Every detection is checked against patches built with a known geometry: a lattice of
// control points on a box, translated so that two boxes share a face, with the neighbour's control
// net then re-indexed by hand (axes swapped, a direction reversed) so the expected `axis_map` and
// `flips` follow from how it was built, not from the code under test. The index arithmetic of
// `match_face_cps` is checked against explicit nested loops over all faces and orientations, and
// against hand-derived id lists. Messages are matched by substring.
//
// Tolerances. Every geometry here is built so that shared coordinates are bitwise equal (a lattice
// plus an integer offset), or displaced by a stated multiple of the unit roundoff of the storage
// type; nothing depends on a measured threshold. The thread-count test compares complete results
// for equality, which the loop's fixed chunks make a property of the algorithm (disjoint
// per-item outputs concatenated in index order).
#include <doctest/doctest.h>

#include "bspline_test_support.hpp"

#include <pantr/bspline/bspline.hpp>
#include <pantr/bspline/convert.hpp>
#include <pantr/core/errors.hpp>
#include <pantr/multipatch/detect.hpp>
#include <pantr/multipatch/interface.hpp>
#include <pantr/multipatch/match.hpp>
#include <pantr/multipatch/multi_patch.hpp>
#include <pantr/parallel/pool.hpp>
#include <pantr/tolerance.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace bs = pantr::bspline;
namespace mp = pantr::multipatch;
namespace par = pantr::parallel;
using bspline_test::clamped;
using bspline_test::Dir;
using bspline_test::message_of;

namespace {

using Idx = std::vector<std::size_t>;
/// The coordinates (the weight last, for a rational patch) of the control point at a multi-index,
/// given the basis counts.
using CoordsAt = std::function<std::vector<double>(const Idx& idx, const Idx& num_basis)>;

/// Restores the calling thread's parallelism setting on scope exit.
class ThreadsGuard {
 public:
  explicit ThreadsGuard(std::size_t n) : previous_(par::get_num_threads()) {
    par::set_num_threads(static_cast<std::int64_t>(n));
  }
  ~ThreadsGuard() { par::set_num_threads(static_cast<std::int64_t>(previous_)); }
  ThreadsGuard(const ThreadsGuard&) = delete;
  ThreadsGuard& operator=(const ThreadsGuard&) = delete;

 private:
  std::size_t previous_;
};

/// A B-spline over `dirs` whose control point at each multi-index is `at(idx, num_basis)`.
template <class T>
bs::Bspline<T> build(const std::vector<Dir>& dirs, const CoordsAt& at, bool rational = false) {
  bs::BsplineSpace<T> space = bspline_test::space_of<T>(dirs);
  Idx nb;
  std::size_t total = 1;
  for (const auto& s : space.spaces()) {
    nb.push_back(static_cast<std::size_t>(s.num_basis()));
    total *= nb.back();
  }
  std::vector<T> cps;
  Idx idx(nb.size(), 0);
  for (std::size_t n = 0; n < total; ++n) {
    for (const double v : at(idx, nb)) {
      cps.push_back(static_cast<T>(v));
    }
    for (std::size_t d = idx.size(); d-- > 0;) {
      if (++idx[d] < nb[d]) {
        break;
      }
      idx[d] = 0;
    }
  }
  return bs::Bspline<T>(std::move(space), std::span<const T>(cps), rational);
}

/// The lattice patch: control point `i` of direction `d` at `offset[d] + i / (n_d - 1)`, so the
/// patch is the box `offset + [0, 1]^dim`, with the control points equally spaced.
template <class T>
bs::Bspline<T> lattice(const std::vector<Dir>& dirs, const std::vector<double>& offset) {
  return build<T>(dirs, [&](const Idx& idx, const Idx& nb) {
    std::vector<double> x(idx.size());
    for (std::size_t d = 0; d < idx.size(); ++d) {
      x[d] = offset[d] + static_cast<double>(idx[d]) / static_cast<double>(nb[d] - 1);
    }
    return x;
  });
}

/// Degree 2, one interior knot: 4 basis functions.
Dir dir4() { return Dir{clamped(2, {0.5}), 2}; }
/// Degree 2, no interior knot: 3 basis functions.
Dir dir3() { return Dir{clamped(2, {}), 2}; }
/// Degree 1, no interior knot: 2 basis functions.
Dir dir2() { return Dir{clamped(1, {}), 1}; }

/// The anisotropic 2D layout, `(4, 3)` basis functions.
std::vector<Dir> dirs_2d() { return {dir4(), dir3()}; }
/// The anisotropic 3D layout, `(2, 3, 4)` basis functions.
std::vector<Dir> dirs_3d() { return {dir2(), dir3(), dir4()}; }

template <class T>
std::span<const bs::Bspline<T>> view(const std::vector<bs::Bspline<T>>& v) {
  return std::span<const bs::Bspline<T>>(v);
}

mp::Interface iface(long long pa, long long fa, long long pb, long long fb,
                    std::vector<long long> axis_map, std::vector<bool> flips) {
  return mp::Interface(pa, fa, pb, fb, std::move(axis_map), std::move(flips));
}

std::string tuple_text(const std::vector<mp::Interface>& v) {
  std::string out;
  for (const auto& i : v) {
    out += i.repr() + "\n";
  }
  return out;
}

}  // namespace

// ------------------------------------------------------------------------------------ Interface

TEST_CASE("multipatch: face ids split into axis and side, and the tangential axes follow") {
  CHECK(mp::face_axis_side(0) == std::pair<long long, long long>{0, 0});
  CHECK(mp::face_axis_side(1) == std::pair<long long, long long>{0, 1});
  CHECK(mp::face_axis_side(4) == std::pair<long long, long long>{2, 0});
  CHECK(mp::face_axis_side(5) == std::pair<long long, long long>{2, 1});
  CHECK(mp::tangential_axes(1, 1).empty());
  CHECK(mp::tangential_axes(3, 2) == std::vector<long long>{0});
  CHECK(mp::tangential_axes(0, 3) == std::vector<long long>{1, 2});
  CHECK(mp::tangential_axes(2, 3) == std::vector<long long>{0, 2});
  CHECK(mp::tangential_axes(5, 3) == std::vector<long long>{0, 1});
}

TEST_CASE("multipatch: an interface keeps its fields and spells itself as the dataclass does") {
  const mp::Interface a = iface(0, 1, 1, 0, {1}, {false});
  CHECK(a.patch_a() == 0);
  CHECK(a.face_a() == 1);
  CHECK(a.patch_b() == 1);
  CHECK(a.face_b() == 0);
  CHECK(a.axis_map() == std::vector<long long>{1});
  CHECK(a.flips() == std::vector<bool>{false});
  CHECK(a.dim() == 2);
  CHECK(a.repr() ==
        "Interface(patch_a=0, face_a=1, patch_b=1, face_b=0, axis_map=(1,), flips=(False,))");
  CHECK(
      iface(2, 5, 3, 2, {0, 2}, {true, false}).repr() ==
      "Interface(patch_a=2, face_a=5, patch_b=3, face_b=2, axis_map=(0, 2), flips=(True, False))");
  CHECK(iface(0, 1, 1, 0, {}, {}).repr() ==
        "Interface(patch_a=0, face_a=1, patch_b=1, face_b=0, axis_map=(), flips=())");
  CHECK(iface(0, 1, 1, 0, {}, {}).dim() == 1);
  CHECK(a == iface(0, 1, 1, 0, {1}, {false}));
  CHECK_FALSE(a == iface(0, 1, 1, 0, {1}, {true}));
  CHECK_FALSE(a == iface(0, 1, 2, 0, {1}, {false}));
}

TEST_CASE(
    "multipatch: an interface that cannot hold is refused, with the first message that fits") {
  using pantr::value_error;
  CHECK_THROWS_AS((void)iface(0, 1, 1, 0, {1}, {}), value_error);
  CHECK(message_of([] { (void)iface(0, 1, 1, 0, {1}, {}); }) ==
        "flips and axis_map must have equal length; got 0 and 1");
  CHECK(message_of([] { (void)iface(0, 4, 1, 0, {1}, {false}); }) ==
        "face_a=4 out of range for dim=2; expected 0 <= face < 4");
  CHECK(message_of([] { (void)iface(0, 1, 1, -1, {1}, {false}); }) ==
        "face_b=-1 out of range for dim=2; expected 0 <= face < 4");
  // The equal-length check precedes the range check, which precedes the self check.
  CHECK(message_of([] { (void)iface(0, 9, 0, 9, {1}, {}); }).find("equal length") !=
        std::string::npos);
  CHECK(message_of([] { (void)iface(0, 9, 0, 9, {1}, {false}); }).find("face_a=9") !=
        std::string::npos);
  CHECK(message_of([] { (void)iface(0, 1, 0, 1, {1}, {false}); }) ==
        "An interface cannot join a face to itself (patch 0, face 1)");
  CHECK(message_of([] { (void)iface(0, 1, 1, 0, {0}, {false}); }) ==
        "axis_map=(0,) must be a permutation of the tangential axes (1,) of face_b=0");
  CHECK(message_of([] { (void)iface(0, 1, 1, 0, {1, 1}, {false, false}); }) ==
        "axis_map=(1, 1) must be a permutation of the tangential axes (1, 2) of face_b=0");
  CHECK(message_of([] { (void)iface(0, 1, 1, 0, {1, 1, 1}, {false, false, false}); }) ==
        "axis_map=(1, 1, 1) must be a permutation of the tangential axes (1, 2, 3) of face_b=0");
  CHECK(message_of([] { (void)iface(1, 0, 0, 1, {1}, {false}); }) ==
        "Interface sides must be in canonical order: (1, 0) must be lexicographically smaller "
        "than (0, 1)");
  // The same patch on two faces is canonical when the faces ascend, not otherwise.
  CHECK_NOTHROW((void)iface(0, 0, 0, 1, {1}, {false}));
  CHECK_THROWS_AS((void)iface(0, 1, 0, 0, {1}, {false}), value_error);
}

// ------------------------------------------------------------------------------- match_face_cps

namespace {

/// The pairing by explicit loops: an independent oracle that walks the tangential index space of
/// face a in C order and computes both flat ids by hand.
mp::MatchedCps brute_force_match(const Idx& n_a, long long face_a, const Idx& n_b, long long face_b,
                                 const std::vector<long long>& axis_map,
                                 const std::vector<bool>& flips) {
  const std::size_t dim = n_a.size();
  const auto axis_a = static_cast<std::size_t>(face_a / 2);
  const auto axis_b = static_cast<std::size_t>(face_b / 2);
  std::vector<std::size_t> tang;
  for (std::size_t k = 0; k < dim; ++k) {
    if (k != axis_a) {
      tang.push_back(k);
    }
  }
  mp::MatchedCps out;
  Idx combo(tang.size(), 0);
  std::size_t count = 1;
  for (const std::size_t axis : tang) {
    count *= n_a[axis];
  }
  for (std::size_t n = 0; n < count; ++n) {
    Idx index_a(dim, 0);
    Idx index_b(dim, 0);
    index_a[axis_a] = face_a % 2 == 0 ? 0 : n_a[axis_a] - 1;
    index_b[axis_b] = face_b % 2 == 0 ? 0 : n_b[axis_b] - 1;
    for (std::size_t k = 0; k < tang.size(); ++k) {
      index_a[tang[k]] = combo[k];
      const auto target = static_cast<std::size_t>(axis_map[k]);
      index_b[target] = flips[k] ? n_b[target] - 1 - combo[k] : combo[k];
    }
    std::size_t flat_a = 0;
    std::size_t flat_b = 0;
    for (std::size_t d = 0; d < dim; ++d) {
      flat_a = flat_a * n_a[d] + index_a[d];
      flat_b = flat_b * n_b[d] + index_b[d];
    }
    out.a.push_back(static_cast<std::int64_t>(flat_a));
    out.b.push_back(static_cast<std::int64_t>(flat_b));
    for (std::size_t k = tang.size(); k-- > 0;) {
      if (++combo[k] < n_a[tang[k]]) {
        break;
      }
      combo[k] = 0;
    }
  }
  return out;
}

}  // namespace

TEST_CASE("match_face_cps: the ids come in the C-order of face a, matched node by node") {
  // 2D: a has (3, 2) functions, face 1 is its high-axis-0 side, whose tangential axis 1 holds 2
  // nodes with flat ids 4, 5 (strides (2, 1), slab 2). b has (2, 2); its face 0 holds ids 0, 1.
  const std::vector<std::size_t> n_a = {3, 2};
  const std::vector<std::size_t> n_b = {2, 2};
  mp::MatchedCps m = mp::match_face_cps(n_a, 1, n_b, 0, std::vector<long long>{1}, {false});
  CHECK(m.a == std::vector<std::int64_t>{4, 5});
  CHECK(m.b == std::vector<std::int64_t>{0, 1});
  m = mp::match_face_cps(n_a, 1, n_b, 0, std::vector<long long>{1}, {true});
  CHECK(m.a == std::vector<std::int64_t>{4, 5});
  CHECK(m.b == std::vector<std::int64_t>{1, 0});

  // 3D with a swap and a flip. a: (2, 3, 4), face 1 (x high, flat ids 12 + 4 j + k over the
  // tangential axes (1, 2), the first one slowest). b: (4, 2, 3), face 2 (axis 1 low): a's axis 1
  // (3 nodes) is b's axis 2, a's axis 2 (4 nodes) is b's axis 0 reversed. b's strides are
  // (6, 3, 1), so the node (j, k) is `(3 - k) * 6 + j`.
  const std::vector<std::size_t> n3_a = {2, 3, 4};
  const std::vector<std::size_t> n3_b = {4, 2, 3};
  m = mp::match_face_cps(n3_a, 1, n3_b, 2, std::vector<long long>{2, 0}, {false, true});
  CHECK(m.a == std::vector<std::int64_t>{12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23});
  CHECK(m.b == std::vector<std::int64_t>{18, 12, 6, 0, 19, 13, 7, 1, 20, 14, 8, 2});

  // 1D: a face is a point; the ids are the end functions.
  const std::vector<std::size_t> n1 = {5};
  m = mp::match_face_cps(n1, 1, n1, 0, std::vector<long long>{}, {});
  CHECK(m.a == std::vector<std::int64_t>{4});
  CHECK(m.b == std::vector<std::int64_t>{0});
}

TEST_CASE("match_face_cps: every face pair and orientation agrees with nested loops") {
  const std::array<Idx, 3> counts = {Idx{4}, Idx{4, 3}, Idx{2, 3, 4}};
  for (const Idx& n : counts) {
    const std::size_t dim = n.size();
    const auto n_faces = static_cast<long long>(2 * dim);
    for (long long face_a = 0; face_a < n_faces; ++face_a) {
      for (long long face_b = 0; face_b < n_faces; ++face_b) {
        std::vector<long long> perm = mp::tangential_axes(face_b, dim);
        do {
          for (unsigned bits = 0; bits < (1U << perm.size()); ++bits) {
            std::vector<bool> flips(perm.size());
            for (std::size_t k = 0; k < perm.size(); ++k) {
              flips[k] = ((bits >> k) & 1U) != 0;
            }
            const std::vector<long long> tang_a = mp::tangential_axes(face_a, dim);
            bool sizes_agree = true;
            for (std::size_t k = 0; k < perm.size(); ++k) {
              sizes_agree = sizes_agree && n[static_cast<std::size_t>(tang_a[k])] ==
                                               n[static_cast<std::size_t>(perm[k])];
            }
            if (!sizes_agree) {
              CHECK(message_of([&] {
                      (void)mp::match_face_cps(n, face_a, n, face_b, perm, flips);
                    }).find("different numbers of basis functions") != std::string::npos);
              continue;
            }
            const mp::MatchedCps got = mp::match_face_cps(n, face_a, n, face_b, perm, flips);
            const mp::MatchedCps want = brute_force_match(n, face_a, n, face_b, perm, flips);
            CHECK(got.a == want.a);
            CHECK(got.b == want.b);
            CHECK(std::is_sorted(got.a.begin(), got.a.end()));
          }
        } while (std::next_permutation(perm.begin(), perm.end()));
      }
    }
  }
}

TEST_CASE("match_face_cps: refusals and their messages, in order") {
  const std::vector<std::size_t> n_2d = {4, 3};
  const std::vector<std::size_t> n_3d = {2, 3, 4};
  const std::vector<long long> one = {1};
  const std::vector<bool> no_flip = {false};
  CHECK(message_of([&] { (void)mp::match_face_cps(n_2d, 1, n_3d, 0, one, no_flip); }) ==
        "Both spaces must have the same parametric dimension; got 2 and 3");
  CHECK(message_of([&] { (void)mp::match_face_cps(n_2d, 4, n_2d, 0, one, no_flip); }) ==
        "face_a=4 out of range for a 2D space; expected 0 <= face_a < 4");
  CHECK(message_of([&] { (void)mp::match_face_cps(n_2d, 0, n_2d, -1, one, no_flip); }) ==
        "face_b=-1 out of range for a 2D space; expected 0 <= face_b < 4");
  CHECK(message_of([&] {
          (void)mp::match_face_cps(n_2d, 1, n_2d, 0, one, std::vector<bool>{false, false});
        }) == "flips and axis_map must have equal length; got 2 and 1");
  CHECK(message_of([&] {
          (void)mp::match_face_cps(n_2d, 1, n_2d, 0, std::vector<long long>{1, 0},
                                   std::vector<bool>{false, false});
        }) == "axis_map must have length 1 for a 2D space");
  CHECK(message_of([&] {
          (void)mp::match_face_cps(n_2d, 1, n_2d, 0, std::vector<long long>{0}, no_flip);
        }) == "axis_map=(0,) must be a permutation of the tangential axes (1,) of face_b=0");
  // Face 1 of the (4, 3) space has 3 nodes along axis 1; face 2 of the same space is matched with
  // axis 0, which has 4.
  CHECK(message_of([&] {
          (void)mp::match_face_cps(n_2d, 1, n_2d, 3, std::vector<long long>{0}, no_flip);
        }) ==
        "Matched tangential directions hold different numbers of basis functions: axis 1 of "
        "space_a has 3, axis 0 of space_b has 4");
}

TEST_CASE("match_face_cps: spaces of different storage types give the id lists of their counts") {
  const bs::BsplineSpace<double> sd = bspline_test::space_of<double>(dirs_2d());
  const bs::BsplineSpace<float> sf = bspline_test::space_of<float>(dirs_2d());
  const mp::MatchedCps by_spaces =
      mp::match_face_cps(sd, 1, sf, 0, std::vector<long long>{1}, {true});
  const mp::MatchedCps by_counts =
      mp::match_face_cps(std::vector<std::size_t>{4, 3}, 1, std::vector<std::size_t>{4, 3}, 0,
                         std::vector<long long>{1}, {true});
  CHECK(by_spaces.a == by_counts.a);
  CHECK(by_spaces.b == by_counts.b);
  CHECK(by_spaces.a == std::vector<std::int64_t>{9, 10, 11});
  CHECK(by_spaces.b == std::vector<std::int64_t>{2, 1, 0});
}

// ------------------------------------------------------------------------------------ detection

TEST_CASE_TEMPLATE("detect: patches side by side in 1D, 2D and 3D share the expected face", T,
                   double, float) {
  {
    const std::vector<Dir> d = {dir3()};
    const std::vector<bs::Bspline<T>> p = {lattice<T>(d, {0.0}), lattice<T>(d, {1.0})};
    CHECK(mp::detect_interfaces(view(p)) == std::vector<mp::Interface>{iface(0, 1, 1, 0, {}, {})});
  }
  {
    const std::vector<bs::Bspline<T>> p = {lattice<T>(dirs_2d(), {0.0, 0.0}),
                                           lattice<T>(dirs_2d(), {1.0, 0.0})};
    CHECK(mp::detect_interfaces(view(p)) ==
          std::vector<mp::Interface>{iface(0, 1, 1, 0, {1}, {false})});
  }
  for (std::size_t axis = 0; axis < 3; ++axis) {
    std::vector<double> offset = {0.0, 0.0, 0.0};
    offset[axis] = 1.0;
    const std::vector<bs::Bspline<T>> p = {lattice<T>(dirs_3d(), {0.0, 0.0, 0.0}),
                                           lattice<T>(dirs_3d(), offset)};
    std::vector<long long> tangential;
    for (long long k = 0; k < 3; ++k) {
      if (k != static_cast<long long>(axis)) {
        tangential.push_back(k);
      }
    }
    const auto a = static_cast<long long>(axis);
    CHECK(mp::detect_interfaces(view(p)) ==
          std::vector<mp::Interface>{iface(0, 2 * a + 1, 1, 2 * a, tangential, {false, false})});
  }
}

TEST_CASE_TEMPLATE("detect: an axis swap and a reversed direction are recovered in 2D", T, double,
                   float) {
  const bs::Bspline<T> first = lattice<T>(dirs_2d(), {0.0, 0.0});
  // The neighbour on the right, with its axes swapped: its index (a, b) is the lattice's (b, a).
  const std::vector<Dir> swapped = {dir3(), dir4()};
  const auto swapped_patch = [&](bool reverse_axis_0) {
    return build<T>(swapped, [&](const Idx& idx, const Idx& nb) {
      const std::size_t j = reverse_axis_0 ? nb[0] - 1 - idx[0] : idx[0];  // the lattice's y index
      const std::size_t i = idx[1];                                        // the lattice's x index
      return std::vector<double>{1.0 + static_cast<double>(i) / 3.0, static_cast<double>(j) / 2.0};
    });
  };
  {
    const std::vector<bs::Bspline<T>> p = {first, swapped_patch(false)};
    CHECK(mp::detect_interfaces(view(p)) ==
          std::vector<mp::Interface>{iface(0, 1, 1, 2, {0}, {false})});
  }
  {
    const std::vector<bs::Bspline<T>> p = {first, swapped_patch(true)};
    CHECK(mp::detect_interfaces(view(p)) ==
          std::vector<mp::Interface>{iface(0, 1, 1, 2, {0}, {true})});
  }
}

TEST_CASE_TEMPLATE("detect: a swap and a flip in 3D, and the paired control points coincide", T,
                   double, float) {
  const bs::Bspline<T> first = lattice<T>(dirs_3d(), {0.0, 0.0, 0.0});
  // The neighbour on the +x side, with index (a, b, c) = (z, x, reversed y): directions z (4
  // functions), x (2), y (3).
  const std::vector<Dir> dirs = {dir4(), dir2(), dir3()};
  const bs::Bspline<T> second = build<T>(dirs, [](const Idx& idx, const Idx& nb) {
    return std::vector<double>{1.0 + static_cast<double>(idx[1]) / 1.0,
                               static_cast<double>(nb[2] - 1 - idx[2]) / 2.0,
                               static_cast<double>(idx[0]) / 3.0};
  });
  const std::vector<bs::Bspline<T>> p = {first, second};
  const std::vector<mp::Interface> found = mp::detect_interfaces(view(p));
  const std::vector<mp::Interface> want = {iface(0, 1, 1, 2, {2, 0}, {true, false})};
  REQUIRE_MESSAGE(found == want, tuple_text(found));

  // The paired control points coincide: the oracle for the whole chain, independent of the record.
  const mp::MatchedCps m =
      mp::match_face_cps(first.space(), found[0].face_a(), second.space(), found[0].face_b(),
                         found[0].axis_map(), found[0].flips());
  REQUIRE(m.a.size() == 12);
  for (std::size_t i = 0; i < m.a.size(); ++i) {
    for (std::size_t c = 0; c < 3; ++c) {
      CHECK(first.control_points()[static_cast<std::size_t>(m.a[i]) * 3 + c] ==
            second.control_points()[static_cast<std::size_t>(m.b[i]) * 3 + c]);
    }
  }
}

TEST_CASE(
    "detect: patches that only touch, or are apart, or differ inside the face, share nothing") {
  const bs::Bspline<double> origin = lattice<double>(dirs_2d(), {0.0, 0.0});
  // Corner contact.
  std::vector<bs::Bspline<double>> p = {origin, lattice<double>(dirs_2d(), {1.0, 1.0})};
  CHECK(mp::detect_interfaces(view(p)).empty());
  // A gap.
  p = {origin, lattice<double>(dirs_2d(), {5.0, 0.0})};
  CHECK(mp::detect_interfaces(view(p)).empty());
  // The corners of the shared face coincide, one interior control point does not: the control
  // points are compared, not only the corners.
  p = {origin, build<double>(dirs_2d(), [](const Idx& idx, const Idx& nb) {
         double y = static_cast<double>(idx[1]) / static_cast<double>(nb[1] - 1);
         if (idx[0] == 0 && idx[1] == 1) {
           y += 0.1;
         }
         return std::vector<double>{
             1.0 + static_cast<double>(idx[0]) / static_cast<double>(nb[0] - 1), y};
       })};
  CHECK(mp::detect_interfaces(view(p)).empty());
}

TEST_CASE("detect: equal geometry with other knots or another degree along the face is refused") {
  const bs::Bspline<double> first = lattice<double>(dirs_2d(), {0.0, 0.0});
  // A refined knot vector on the shared face (axis 1: 4 functions where the first has 3).
  std::vector<bs::Bspline<double>> p = {
      first, lattice<double>({dir4(), Dir{clamped(2, {0.5}), 2}}, {1.0, 0.0})};
  CHECK(mp::detect_interfaces(view(p)).empty());
  // The same count of functions, an interior knot elsewhere: the normalized knots disagree.
  p = {lattice<double>({dir3(), Dir{clamped(2, {0.25}), 2}}, {0.0, 0.0}),
       lattice<double>({dir3(), Dir{clamped(2, {0.75}), 2}}, {1.0, 0.0})};
  CHECK(mp::detect_interfaces(view(p)).empty());
  // Another degree along the face.
  p = {first, lattice<double>({dir4(), Dir{clamped(1, {0.5}), 1}}, {1.0, 0.0})};
  CHECK(mp::detect_interfaces(view(p)).empty());
}

TEST_CASE(
    "detect: knot vectors are compared after a map onto [0, 1], so a rescaled domain agrees") {
  const std::vector<Dir> d_a = {dir3(),
                                Dir{std::vector<double>{0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0}, 2}};
  const std::vector<Dir> d_b = {dir3(),
                                Dir{std::vector<double>{3.0, 3.0, 3.0, 5.0, 7.0, 7.0, 7.0}, 2}};
  const std::vector<bs::Bspline<double>> p = {lattice<double>(d_a, {0.0, 0.0}),
                                              lattice<double>(d_b, {1.0, 0.0})};
  CHECK(mp::detect_interfaces(view(p)) ==
        std::vector<mp::Interface>{iface(0, 1, 1, 0, {1}, {false})});
}

TEST_CASE("detect: when several orientations pass, the first in the enumeration wins") {
  // The shared face collapses to the point (1, 0.5) on both sides, so both orientations match it:
  // the one without a flip comes first. (Patch a's high-x face and patch b's low-x face; the
  // other faces are a unit square's edges, which no other face reproduces.)
  const std::vector<Dir> d = {dir2(), dir3()};
  const auto patch = [&](bool collapse_high) {
    return build<double>(d, [collapse_high](const Idx& idx, const Idx& nb) {
      const double y = static_cast<double>(idx[1]) / static_cast<double>(nb[1] - 1);
      const bool collapsed = collapse_high ? idx[0] == 1 : idx[0] == 0;
      const double x = (collapse_high ? 0.0 : 1.0) + static_cast<double>(idx[0]);
      return std::vector<double>{collapsed ? 1.0 : x, collapsed ? 0.5 : y};
    });
  };
  const std::vector<bs::Bspline<double>> p = {patch(true), patch(false)};
  CHECK(mp::detect_interfaces(view(p)) ==
        std::vector<mp::Interface>{iface(0, 1, 1, 0, {1}, {false})});
}

TEST_CASE(
    "detect: with several passing orientations the permutation and the flip bits order them") {
  // A 3D face whose control points are `(1, u v, 0)` with `u = i - 1`, `v = j - 1` on a 3 x 3 net:
  // symmetric under reversing both axes, not under one. The neighbour's face is the first one's
  // with axis 0 reversed, so the flips `(true, false)` and `(false, true)` both reproduce it and
  // the identity does not. Candidates run through the flip patterns with the first axis the most
  // significant bit, so `(false, true)` is met first.
  const std::vector<Dir> d = {dir2(), dir3(), dir3()};
  const auto patch = [&](bool high_face, double x_shift) {
    return build<double>(d, [=](const Idx& idx, const Idx&) {
      const double u = static_cast<double>(idx[1]) - 1.0;
      const double v = static_cast<double>(idx[2]) - 1.0;
      const bool on_face = high_face ? idx[0] == 1 : idx[0] == 0;
      const double x = x_shift + static_cast<double>(idx[0]);
      // The shared face sits at x = 1; the other face is a plane at x = 0 or x = 2.
      return std::vector<double>{on_face ? 1.0 : x, on_face ? (high_face ? 1.0 : -1.0) * u * v : u,
                                 on_face ? 0.0 : v};
    });
  };
  // The first patch's high face is `u v`; the second's low face is `-u v`, the reversal of axis 0.
  const std::vector<bs::Bspline<double>> p = {patch(true, 0.0), patch(false, 1.0)};
  const std::vector<mp::Interface> found = mp::detect_interfaces(view(p));
  REQUIRE_MESSAGE(found == std::vector<mp::Interface>{iface(0, 1, 1, 0, {1, 2}, {false, true})},
                  tuple_text(found));
}

TEST_CASE("detect: the tolerances are the strict tier, at their scale, to within a small factor") {
  // Double precision: the strict tier is 4 eps, about 8.9e-16, of the scale `max(diagonal,
  // magnitude)`, here the diagonal of the joint box, about 2.24. A displacement of the shared face
  // a few ulps of 1 is within it, one ten times larger (or more) is not. The corners of the face
  // are checked first against the same tolerance, so the interior nodes are displaced alone to
  // reach the comparison of the matched control points.
  const auto pair = [](double gap, double interior_gap, double knot_shift, double weight_bump) {
    const auto make = [&](double x0, bool second) {
      return build<double>(
          {dir3(), Dir{clamped(2, {second ? 0.5 + knot_shift : 0.5}), 2}},
          [=](const Idx& idx, const Idx& nb) {
            const double y = static_cast<double>(idx[1]) / static_cast<double>(nb[1] - 1);
            double w = 1.0 + 0.5 * y;
            if (second && idx[0] == 0 && idx[1] == 0) {
              w += weight_bump;
            }
            const bool corner = idx[1] == 0 || idx[1] + 1 == nb[1];
            const double shift = second && idx[0] == 0 ? (corner ? gap : interior_gap) : 0.0;
            return std::vector<double>{
                x0 + shift + static_cast<double>(idx[0]) / static_cast<double>(nb[0] - 1), y, w};
          },
          true);
    };
    return std::vector<bs::Bspline<double>>{make(0.0, false), make(1.0, true)};
  };
  const std::vector<mp::Interface> one = {iface(0, 1, 1, 0, {1}, {false})};
  // Coordinates: the corners, then the interior nodes.
  auto p = pair(1e-15, 0.0, 0.0, 0.0);
  CHECK(mp::detect_interfaces(view(p)) == one);
  p = pair(1e-14, 0.0, 0.0, 0.0);
  CHECK(mp::detect_interfaces(view(p)).empty());
  p = pair(0.0, 1e-15, 0.0, 0.0);
  CHECK(mp::detect_interfaces(view(p)) == one);
  p = pair(0.0, 1e-14, 0.0, 0.0);
  CHECK(mp::detect_interfaces(view(p)).empty());
  // Knots, after the map onto [0, 1]: the bare strict tier.
  p = pair(0.0, 0.0, 4e-16, 0.0);
  CHECK(mp::detect_interfaces(view(p)) == one);
  p = pair(0.0, 0.0, 1e-14, 0.0);
  CHECK(mp::detect_interfaces(view(p)).empty());
  // Weights: the strict tier times the largest weight in play (about 1.5).
  p = pair(0.0, 0.0, 0.0, 1e-15);
  CHECK(mp::detect_interfaces(view(p)) == one);
  p = pair(0.0, 0.0, 0.0, 1e-13);
  CHECK(mp::detect_interfaces(view(p)).empty());
}

TEST_CASE("detect: rational weights are compared, and the weight scale follows the largest one") {
  const std::vector<Dir> d = {dir3(), dir3()};
  const auto rational_patch = [&](double x0, double bump) {
    return build<double>(
        d,
        [&](const Idx& idx, const Idx& nb) {
          const double y = static_cast<double>(idx[1]) / static_cast<double>(nb[1] - 1);
          double w = 1.0 + 0.5 * y;
          if (x0 > 0.0 && idx[0] == 0 && idx[1] == 0) {
            w += bump;
          }
          return std::vector<double>{
              x0 + static_cast<double>(idx[0]) / static_cast<double>(nb[0] - 1), y, w};
        },
        true);
  };
  // The neighbour's low face copies the first's high face: coordinates are x0 + 0 = 1 and
  // x = 1 for i = 2 (n - 1 = 2), weights are the same expression of y.
  std::vector<bs::Bspline<double>> p = {rational_patch(0.0, 0.0), rational_patch(1.0, 0.0)};
  CHECK(mp::detect_interfaces(view(p)) ==
        std::vector<mp::Interface>{iface(0, 1, 1, 0, {1}, {false})});
  p = {rational_patch(0.0, 0.0), rational_patch(1.0, 0.25)};
  CHECK(mp::detect_interfaces(view(p)).empty());
}

TEST_CASE("detect: a rank mismatch never matches, and a too narrow second patch is refused") {
  const std::vector<Dir> d = {dir3(), dir3()};
  const auto flat = [&](std::size_t rank) {
    return build<double>(d, [rank](const Idx& idx, const Idx& nb) {
      std::vector<double> x(rank, 0.0);
      x[0] = static_cast<double>(idx[0]) / static_cast<double>(nb[0] - 1);
      x[1 % rank] += static_cast<double>(idx[1]) / static_cast<double>(nb[1] - 1);
      return x;
    });
  };
  // Inherited from the Python reference: the joint tolerance reads the first patch's rank columns
  // of both patches. A second patch with at least as many coordinates is read, and its faces
  // differ in rank, so there is no interface.
  std::vector<bs::Bspline<double>> p = {flat(2), flat(3)};
  CHECK(mp::detect_interfaces(view(p)).empty());
  // A second patch with fewer coordinates than the first's rank has nothing to read: refused (the
  // reference died in a NumPy concatenation; the message is this library's own).
  p = {flat(2), flat(1)};
  const std::string msg = message_of([&] { (void)mp::detect_interfaces(view(p)); });
  CHECK(
      msg.find(
          "Patch 1 has 1 control-point coordinates per point, fewer than the rank 2 of patch 0") !=
      std::string::npos);
  // The other order reads one column of each, and the faces differ in rank: no interface between
  // the two patches (the rank-1 patch, a scalar field, meets itself, which is not what is asked).
  p = {flat(1), flat(2)};
  for (const mp::Interface& i : mp::detect_interfaces(view(p))) {
    CHECK(i.patch_a() == i.patch_b());
  }
  // The refusal is the same whichever pair a thread meets first: the first pair in loop order.
  p = {flat(2), flat(2), flat(1), flat(1)};
  CHECK(message_of([&] { (void)mp::detect_interfaces(view(p)); }).find("Patch 2 has 1") !=
        std::string::npos);
}

TEST_CASE(
    "detect: the weight is compared when the first patch is rational, against the last column") {
  // Inherited from the Python reference. Equal rank 2 in both patches, the first rational (x, y,
  // w), the second not (x, y): the last column of the second is the y coordinate.
  const std::vector<Dir> d = {dir3(), dir3()};
  const auto make = [&](double x0, bool rational) {
    return build<double>(
        d,
        [&](const Idx& idx, const Idx& nb) {
          std::vector<double> v = {
              x0 + static_cast<double>(idx[0]) / static_cast<double>(nb[0] - 1),
              static_cast<double>(idx[1]) / static_cast<double>(nb[1] - 1)};
          if (rational) {
            v.push_back(1.0);
          }
          return v;
        },
        rational);
  };
  // Rational first: its weight 1 is compared with the second's y, which is 0 at the first node.
  std::vector<bs::Bspline<double>> p = {make(0.0, true), make(1.0, false)};
  CHECK(mp::detect_interfaces(view(p)).empty());
  // Polynomial first: no weight comparison, and the second's weight column is never read as a
  // coordinate (its first two columns are compared).
  p = {make(0.0, false), make(1.0, true)};
  CHECK(mp::detect_interfaces(view(p)) ==
        std::vector<mp::Interface>{iface(0, 1, 1, 0, {1}, {false})});
}

TEST_CASE("detect: a closed ring is a self-interface, and a periodic direction is not special") {
  // The first direction closes on itself: its end control points coincide.
  const auto ring = [](Dir around) {
    return build<double>({std::move(around), dir2()}, [](const Idx& idx, const Idx& nb) {
      const std::size_t last = nb[0] - 1;
      const std::size_t i = idx[0] == last ? 0 : idx[0];
      return std::vector<double>{static_cast<double>(i), static_cast<double>(idx[1])};
    });
  };
  std::vector<bs::Bspline<double>> p = {ring(Dir{clamped(1, {0.5}), 1})};
  CHECK(mp::detect_interfaces(view(p)) ==
        std::vector<mp::Interface>{iface(0, 0, 0, 1, {1}, {false})});
  // Flagged periodic, the stored control points are compared like any direction's: here again the
  // first and the last coincide.
  p = {ring(Dir{bspline_test::periodic_knots(4, 2), 2, true})};
  CHECK(mp::detect_interfaces(view(p)) ==
        std::vector<mp::Interface>{iface(0, 0, 0, 1, {1}, {false})});
}

TEST_CASE("detect: an L of three patches has two interfaces, in sorted order") {
  const std::vector<Dir> d = {dir3(), dir3()};
  const std::vector<bs::Bspline<double>> p = {lattice<double>(d, {0.0, 0.0}),
                                              lattice<double>(d, {1.0, 0.0}),
                                              lattice<double>(d, {0.0, 1.0})};
  CHECK(
      mp::detect_interfaces(view(p)) ==
      std::vector<mp::Interface>{iface(0, 1, 1, 0, {1}, {false}), iface(0, 3, 2, 2, {0}, {false})});
}

TEST_CASE("detect: no patches give no interfaces, and mixed dimensions are an error") {
  const std::vector<bs::Bspline<double>> none;
  CHECK(mp::detect_interfaces(view(none)).empty());
  const std::vector<bs::Bspline<double>> p = {lattice<double>(dirs_2d(), {0.0, 0.0}),
                                              lattice<double>(dirs_3d(), {0.0, 0.0, 0.0})};
  CHECK(message_of([&] { (void)mp::detect_interfaces(view(p)); }) ==
        "All patches must share one parametric dimension; got [2, 3]");
}

TEST_CASE("detect: the geometric tolerance follows the size of the geometry and where it sits") {
  const std::vector<Dir> d = dirs_2d();
  // Rescaling: a uniform scale of both patches keeps the interface.
  for (const double scale : {1e-4, 1.0, 1e5}) {
    const auto scaled = [&](double x0) {
      return build<double>(d, [&](const Idx& idx, const Idx& nb) {
        return std::vector<double>{
            scale * (x0 + static_cast<double>(idx[0]) / static_cast<double>(nb[0] - 1)),
            scale * static_cast<double>(idx[1]) / static_cast<double>(nb[1] - 1)};
      });
    };
    const std::vector<bs::Bspline<double>> p = {scaled(0.0), scaled(1.0)};
    CHECK_MESSAGE(mp::detect_interfaces(view(p)) ==
                      std::vector<mp::Interface>{iface(0, 1, 1, 0, {1}, {false})},
                  "scale " << scale);
  }
  // Translation: the second patch is displaced by one ulp of the coordinate magnitude, so the
  // coordinate difference carries a rounding at that magnitude, which a tolerance scaled by the
  // diagonal (of order one) alone would not forgive.
  for (const double shift : {0.0, 1e3, 1e6, 1e9}) {
    const double ulp = shift > 0.0 ? std::nextafter(shift, 2.0 * shift) - shift : 0.0;
    const auto moved = [&](double x0, double extra) {
      return build<double>(d, [&](const Idx& idx, const Idx& nb) {
        return std::vector<double>{
            shift + extra + x0 + static_cast<double>(idx[0]) / static_cast<double>(nb[0] - 1),
            shift + extra + static_cast<double>(idx[1]) / static_cast<double>(nb[1] - 1)};
      });
    };
    const std::vector<bs::Bspline<double>> p = {moved(0.0, 0.0), moved(1.0, ulp)};
    CHECK_MESSAGE(mp::detect_interfaces(view(p)) ==
                      std::vector<mp::Interface>{iface(0, 1, 1, 0, {1}, {false})},
                  "shift " << shift);
  }
  // A genuine gap stays a gap far from the origin, where the magnitude term loosens the tolerance.
  const auto far = [&](double x0) {
    return build<double>(d, [&](const Idx& idx, const Idx& nb) {
      return std::vector<double>{
          1e6 + x0 + static_cast<double>(idx[0]) / static_cast<double>(nb[0] - 1),
          1e6 + static_cast<double>(idx[1]) / static_cast<double>(nb[1] - 1)};
    });
  };
  const std::vector<bs::Bspline<double>> apart = {far(0.0), far(2.0)};
  CHECK(mp::detect_interfaces(view(apart)).empty());
}

TEST_CASE("detect: the tolerance tier is the storage type's, and an override replaces it") {
  // The second patch's shared face is displaced by 3e-7: within a few unit roundoffs of float (the
  // strict tier is 4 eps = 4.8e-7 of the scale, about 2) and far beyond double's.
  const double gap = 3e-7;
  const auto pair = [&](auto tag) {
    using T = decltype(tag);
    return std::vector<bs::Bspline<T>>{
        lattice<T>(dirs_2d(), {0.0, 0.0}), build<T>(dirs_2d(), [&](const Idx& idx, const Idx& nb) {
          return std::vector<double>{
              1.0 + (idx[0] == 0 ? gap : 0.0) +
                  static_cast<double>(idx[0]) / static_cast<double>(nb[0] - 1),
              static_cast<double>(idx[1]) / static_cast<double>(nb[1] - 1)};
        })};
  };
  const auto f32 = pair(float{});
  const auto f64 = pair(double{});
  CHECK(mp::detect_interfaces(view(f32)) ==
        std::vector<mp::Interface>{iface(0, 1, 1, 0, {1}, {false})});
  CHECK(mp::detect_interfaces(view(f64)).empty());
  // An override is a relative tolerance in either type.
  CHECK(mp::detect_interfaces(view(f64), 1e-3) ==
        std::vector<mp::Interface>{iface(0, 1, 1, 0, {1}, {false})});
  CHECK(mp::detect_interfaces(view(f32), 1e-12).empty());
  // A NaN override matches nothing.
  CHECK(mp::detect_interfaces(view(f64), std::numeric_limits<double>::quiet_NaN()).empty());
}

TEST_CASE(
    "detect: patches converted to one type compare at the larger relative tolerance of a pair") {
  // The detail overload the binding uses for patches of different dtypes: per-patch relative
  // tolerances, a pair at the larger of its two.
  const double gap = 1e-6;
  const std::vector<bs::Bspline<double>> p = {
      lattice<double>(dirs_2d(), {0.0, 0.0}),
      build<double>(dirs_2d(), [&](const Idx& idx, const Idx& nb) {
        return std::vector<double>{1.0 + (idx[0] == 0 ? gap : 0.0) +
                                       static_cast<double>(idx[0]) / static_cast<double>(nb[0] - 1),
                                   static_cast<double>(idx[1]) / static_cast<double>(nb[1] - 1)};
      })};
  const double strict = static_cast<double>(pantr::tolerance::get_strict<double>());
  const double loose = static_cast<double>(pantr::tolerance::get_strict<float>());
  const std::vector<double> both_strict = {strict, strict};
  const std::vector<double> one_loose = {strict, loose};
  CHECK(mp::detail::detect_interfaces<double>(view(p), both_strict, std::nullopt).empty());
  CHECK(mp::detail::detect_interfaces<double>(view(p), one_loose, std::nullopt) ==
        std::vector<mp::Interface>{iface(0, 1, 1, 0, {1}, {false})});
  // An explicit tolerance replaces both.
  CHECK(mp::detail::detect_interfaces<double>(view(p), one_loose, 1e-12).empty());
}

TEST_CASE("detect: the result does not depend on the number of threads") {
  // A grid of unit boxes, 5 per side: 125 patches, 750 sides, so the loop is above the work model's
  // threshold and runs on the pool. The interfaces follow from the construction: each patch meets
  // its neighbor along +x, +y and +z.
  constexpr std::size_t n = 5;
  const std::vector<Dir> d = {dir2(), dir2(), dir2()};
  std::vector<bs::Bspline<double>> p;
  for (std::size_t ix = 0; ix < n; ++ix) {
    for (std::size_t iy = 0; iy < n; ++iy) {
      for (std::size_t iz = 0; iz < n; ++iz) {
        p.push_back(lattice<double>(
            d, {static_cast<double>(ix), static_cast<double>(iy), static_cast<double>(iz)}));
      }
    }
  }
  std::vector<mp::Interface> want;
  for (std::size_t ix = 0; ix < n; ++ix) {
    for (std::size_t iy = 0; iy < n; ++iy) {
      for (std::size_t iz = 0; iz < n; ++iz) {
        const auto id = static_cast<long long>((ix * n + iy) * n + iz);
        if (ix + 1 < n) {
          want.push_back(iface(id, 1, id + 25, 0, {1, 2}, {false, false}));
        }
        if (iy + 1 < n) {
          want.push_back(iface(id, 3, id + 5, 2, {0, 2}, {false, false}));
        }
        if (iz + 1 < n) {
          want.push_back(iface(id, 5, id + 1, 4, {0, 1}, {false, false}));
        }
      }
    }
  }
  std::sort(want.begin(), want.end(), [](const mp::Interface& x, const mp::Interface& y) {
    return std::make_tuple(x.patch_a(), x.face_a(), x.patch_b(), x.face_b()) <
           std::make_tuple(y.patch_a(), y.face_a(), y.patch_b(), y.face_b());
  });
  REQUIRE(want.size() == 300);

  std::vector<mp::Interface> serial;
  {
    const ThreadsGuard one(1);
    serial = mp::detect_interfaces(view(p));
  }
  CHECK(serial == want);
  std::vector<mp::Interface> many;
  {
    const ThreadsGuard several((std::min<std::size_t>)(8, par::max_threads()));
    many = mp::detect_interfaces(view(p));
  }
  CHECK(many == want);
}

// ----------------------------------------------------------------------------- verify_interface

TEST_CASE("verify_interface: an interface holds when its pairing and knots do") {
  const std::vector<bs::Bspline<double>> p = {lattice<double>(dirs_2d(), {0.0, 0.0}),
                                              lattice<double>(dirs_2d(), {1.0, 0.0})};
  CHECK(mp::verify_interface(view(p), iface(0, 1, 1, 0, {1}, {false})));
  CHECK_FALSE(mp::verify_interface(view(p), iface(0, 1, 1, 0, {1}, {true})));
  CHECK_FALSE(mp::verify_interface(view(p), iface(0, 3, 1, 2, {0}, {false})));
  // The knots disagree: a face of the first, with a face of another count of functions.
  const std::vector<bs::Bspline<double>> q = {lattice<double>(dirs_2d(), {0.0, 0.0}),
                                              lattice<double>({dir4(), dir4()}, {1.0, 0.0})};
  CHECK_FALSE(mp::verify_interface(view(q), iface(0, 1, 1, 0, {1}, {false})));
  // An override.
  CHECK_FALSE(mp::verify_interface(view(p), iface(0, 1, 1, 0, {1}, {true}), 1e-9));
  CHECK(mp::verify_interface(view(p), iface(0, 1, 1, 0, {1}, {false}), 1e-9));
}

TEST_CASE("verify_interface: refusals and their messages") {
  const std::vector<bs::Bspline<double>> p = {lattice<double>(dirs_2d(), {0.0, 0.0}),
                                              lattice<double>(dirs_2d(), {1.0, 0.0})};
  CHECK_THROWS_AS((void)mp::verify_interface(view(p), iface(0, 1, 7, 0, {1}, {false})),
                  pantr::index_error);
  CHECK(message_of([&] {
          (void)mp::verify_interface(view(p), iface(0, 1, 7, 0, {1}, {false}));
        }).find("patch index 7 is out of range for 2 patches") != std::string::npos);
  CHECK(message_of([&] {
          (void)mp::verify_interface(view(p), iface(0, 1, 1, 0, {1, 2}, {false, false}));
        }) == "Interface implies dim=3 but the patches are 2D");
}

// ------------------------------------------------------------------------------------ MultiPatch

TEST_CASE("MultiPatch: interfaces are detected unless supplied, and supplied ones are verified") {
  const std::vector<Dir> d = dirs_2d();
  const std::vector<bs::Bspline<double>> p = {lattice<double>(d, {0.0, 0.0}),
                                              lattice<double>(d, {1.0, 0.0})};
  const mp::MultiPatch<double> detected(p, std::nullopt, std::nullopt);
  CHECK(detected.size() == 2);
  CHECK(detected.dim() == 2);
  CHECK(detected.patches().size() == 2);
  CHECK(detected.interfaces() == std::vector<mp::Interface>{iface(0, 1, 1, 0, {1}, {false})});

  const std::vector<mp::Interface> supplied = {iface(0, 1, 1, 0, {1}, {false})};
  const mp::MultiPatch<double> kept(p, supplied, std::nullopt);
  CHECK(kept.interfaces() == supplied);
  // A supplied empty list is not a request to detect.
  const mp::MultiPatch<double> none(p, std::vector<mp::Interface>{}, std::nullopt);
  CHECK(none.interfaces().empty());
}

TEST_CASE("MultiPatch: refusals and their messages, in order") {
  const std::vector<Dir> d = dirs_2d();
  const std::vector<bs::Bspline<double>> p = {lattice<double>(d, {0.0, 0.0}),
                                              lattice<double>(d, {1.0, 0.0})};
  using Supplied = std::optional<std::vector<mp::Interface>>;
  const auto message = [](const std::vector<bs::Bspline<double>>& patches, const Supplied& given) {
    return message_of([&] { (void)mp::MultiPatch<double>(patches, given, std::nullopt); });
  };
  CHECK(message({}, std::nullopt) == "A MultiPatch needs at least one patch");

  // The attributes are checked patch by patch, dim before rank before rationality.
  const bs::Bspline<double> solid = lattice<double>(dirs_3d(), {0.0, 0.0, 0.0});
  CHECK(message({p[0], solid}, std::nullopt) ==
        "All patches must share dim; patch 0 has 2 but patch 1 has 3");
  const std::vector<Dir> d2 = {dir3(), dir3()};
  const bs::Bspline<double> spatial =
      build<double>(d2, [](const Idx&, const Idx&) { return std::vector<double>{0.0, 0.0, 0.0}; });
  CHECK(message({lattice<double>(d2, {0.0, 0.0}), spatial}, std::nullopt) ==
        "All patches must share rank; patch 0 has 2 but patch 1 has 3");
  const bs::Bspline<double> rational = build<double>(
      d2, [](const Idx&, const Idx&) { return std::vector<double>{0.0, 0.0, 1.0}; }, true);
  CHECK(message({rational, lattice<double>(d2, {0.0, 0.0})}, std::nullopt) ==
        "All patches must share is_rational; patch 0 has True but patch 1 has False");
  CHECK(message({lattice<double>(d2, {0.0, 0.0}), rational}, std::nullopt) ==
        "All patches must share is_rational; patch 0 has False but patch 1 has True");
  // The first disagreeing patch is the one named.
  CHECK(message({p[0], p[1], solid}, std::nullopt) ==
        "All patches must share dim; patch 0 has 2 but patch 2 has 3");
}

TEST_CASE(
    "MultiPatch: a supplied interface is refused when it names no patch, no dimension or no "
    "truth") {
  const std::vector<Dir> d = dirs_2d();
  const std::vector<bs::Bspline<double>> p = {lattice<double>(d, {0.0, 0.0}),
                                              lattice<double>(d, {1.0, 0.0})};
  const auto message = [&](const mp::Interface& i) {
    return message_of(
        [&] { (void)mp::MultiPatch<double>(p, std::vector<mp::Interface>{i}, std::nullopt); });
  };
  CHECK(message(iface(0, 1, 7, 0, {1}, {false})) ==
        "Interface patch_b=7 is out of range for 2 patches");
  CHECK(message(iface(5, 1, 7, 0, {1}, {false})) ==
        "Interface patch_a=5 is out of range for 2 patches");
  CHECK(message(iface(0, 1, 1, 0, {1, 2}, {false, false})) ==
        "Interface implies dim=3 but the patches are 2D");
  CHECK(message(iface(0, 1, 1, 0, {1}, {true})) ==
        "Interface does not hold for the given geometry: Interface(patch_a=0, face_a=1, "
        "patch_b=1, face_b=0, axis_map=(1,), flips=(True,)). The matched control points or the "
        "tangential knot vectors disagree.");
}

// ------------------------------------------------------------------------------------- astype

TEST_CASE("astype: narrowing casts knots and control points and keeps the rest") {
  // A periodic direction and a direction with a knot that is not a float; rational.
  const std::vector<Dir> dirs = {Dir{bspline_test::periodic_knots(4, 2), 2, true},
                                 Dir{clamped(2, {0.1}), 2}};
  const bs::Bspline<double> b = bspline_test::make_spline<double>(dirs, 2, true, 7);
  const bs::Bspline<float> f = bs::astype<float>(b);
  CHECK(f.dim() == 2);
  CHECK(f.rational());
  CHECK(f.rank() == 2);
  CHECK(f.cp_size() == 3);
  CHECK(f.degrees() == b.degrees());
  CHECK(f.shape()[0] == b.shape()[0]);
  CHECK(f.shape()[1] == b.shape()[1]);
  const bs::BsplineSpace<double> sd = b.space();
  const bs::BsplineSpace<float> sf = f.space();
  for (std::size_t d = 0; d < 2; ++d) {
    CHECK(sf.spaces()[d].periodic() == sd.spaces()[d].periodic());
    CHECK(sf.spaces()[d].degree() == sd.spaces()[d].degree());
    REQUIRE(sf.spaces()[d].knots().size() == sd.spaces()[d].knots().size());
    for (std::size_t i = 0; i < sd.spaces()[d].knots().size(); ++i) {
      CHECK(sf.spaces()[d].knots()[i] == static_cast<float>(sd.spaces()[d].knots()[i]));
    }
  }
  REQUIRE(f.control_points().size() == b.control_points().size());
  for (std::size_t i = 0; i < f.control_points().size(); ++i) {
    CHECK(f.control_points()[i] == static_cast<float>(b.control_points()[i]));
  }
}

TEST_CASE("astype: widening is exact and round-trips") {
  const std::vector<Dir> dirs = {Dir{clamped(2, {0.1}), 2}, Dir{clamped(1, {}), 1}};
  const bs::Bspline<float> f = bspline_test::make_spline<float>(dirs, 3, false, 3);
  const bs::Bspline<double> d = bs::astype<double>(f);
  REQUIRE(d.control_points().size() == f.control_points().size());
  for (std::size_t i = 0; i < f.control_points().size(); ++i) {
    CHECK(d.control_points()[i] == static_cast<double>(f.control_points()[i]));
  }
  const bs::Bspline<float> back = bs::astype<float>(d);
  for (std::size_t i = 0; i < f.control_points().size(); ++i) {
    CHECK(back.control_points()[i] == f.control_points()[i]);
  }
  const bs::BsplineSpace<double> space_d = d.space();
  const bs::BsplineSpace<float> space_f = f.space();
  for (std::size_t k = 0; k < 2; ++k) {
    const auto kd = space_d.spaces()[k].knots();
    const auto kf = space_f.spaces()[k].knots();
    for (std::size_t i = 0; i < kf.size(); ++i) {
      CHECK(kd[i] == static_cast<double>(kf[i]));
    }
  }
}

TEST_CASE("astype: the same type is the same value, and a value beyond the range is refused") {
  const std::vector<Dir> dirs = {Dir{clamped(1, {}), 1}};
  const bs::Bspline<double> b = bspline_test::make_spline<double>(dirs, 1, false, 1);
  CHECK(bs::astype<double>(b).same_rep(b));

  const std::vector<double> huge = {0.0, 1e300};
  const bs::Bspline<double> big(bspline_test::space_of<double>(dirs), std::span<const double>(huge),
                                false);
  CHECK(message_of([&] { (void)bs::astype<float>(big); }) ==
        "The value 1e+300 is outside the range of the target floating-point type.");
  const std::vector<double> inf = {0.0, std::numeric_limits<double>::infinity()};
  const bs::Bspline<float> kept = bs::astype<float>(bs::Bspline<double>(
      bspline_test::space_of<double>(dirs), std::span<const double>(inf), false));
  CHECK(std::isinf(kept.control_points()[1]));
  // A knot beyond the range.
  const bs::Bspline<double> wide(
      bspline_test::space_of<double>({Dir{std::vector<double>{0.0, 0.0, 1e300, 1e300}, 1}}),
      std::span<const double>(huge), false);
  CHECK_THROWS_AS((void)bs::astype<float>(wide), pantr::value_error);
}
