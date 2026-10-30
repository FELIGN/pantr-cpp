// BsplineSpace1D and BsplineSpace: construction, derived state, and the basis at points. Oracles:
// closed-form low-degree B-splines, partition of unity, and the Python suite's documented examples.
#include <doctest/doctest.h>

#include <pantr/bspline/space.hpp>
#include <pantr/bspline/space_1d.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace bs = pantr::bspline;

TEST_CASE("BsplineSpace1D: derived state of an open quadratic space") {
  const std::vector<double> knots{0, 0, 0, 1, 2, 2, 3, 3, 3};
  const bs::BsplineSpace1D<double> s(knots, 2);
  CHECK(s.num_basis() == 6);
  CHECK(s.num_intervals() == 3);
  const auto first = s.first_basis_per_interval();
  CHECK(std::vector<std::int64_t>(first.begin(), first.end()) ==
        std::vector<std::int64_t>{0, 1, 3});
  CHECK(s.has_open_knots());
  CHECK_FALSE(s.has_bezier_like_knots());
}

TEST_CASE("BsplineSpace1D: tabulate_basis matches the documented example") {
  const std::vector<double> knots{0, 0, 0, 0.25, 0.7, 0.7, 1, 1, 1};
  const bs::BsplineSpace1D<double> s(knots, 2);
  const std::vector<double> pts{0.0, 0.5, 0.75, 1.0};
  auto [v, f] = s.tabulate_basis<double>(pts);
  const double expected[4][3] = {{1.0, 0.0, 0.0},
                                 {0.12698413, 0.5643739, 0.30864198},
                                 {0.69444444, 0.27777778, 0.02777778},
                                 {0.0, 0.0, 1.0}};
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 3; ++j) {
      CHECK(v.data()[i * 3 + j] == doctest::Approx(expected[i][j]).epsilon(1e-7));
    }
  }
  CHECK(std::vector<std::int64_t>(f.data(), f.data() + 4) == std::vector<std::int64_t>{0, 1, 3, 3});
}

TEST_CASE("BsplineSpace: lattice tensor product") {
  const std::vector<double> knots{0, 0, 0, 1, 2, 2, 2};
  const bs::BsplineSpace1D<double> s(knots, 2);
  const bs::BsplineSpace<double> sp({s, s});
  const pantr::quad::PointsLattice<double> lat({{0.5, 1.5}, {0.25}});
  pantr::NdArray<double> out({2, 9}, pantr::for_overwrite);
  pantr::NdArray<std::int64_t> first({2, 2}, pantr::for_overwrite);
  sp.tabulate_basis<double>(lat, out.view<2>(), first.view<2>());
  double sum = 0;
  for (std::size_t j = 0; j < 9; ++j) {
    sum += out.data()[j];
  }
  CHECK(sum == doctest::Approx(1.0));
}
