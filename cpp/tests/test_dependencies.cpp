// The pinned dependencies work together: Spectra 1.2.0 on the pinned Eigen. This is the check
// that chose Eigen 5.0.1 over 3.4.1 (cmake/PantrDependencies.cmake).
#include <doctest/doctest.h>

#include <pantr/pantr.hpp>

#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/SymEigsSolver.h>
#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <cmath>
#include <numbers>
#include <string>
#include <vector>

TEST_CASE("dependencies: Eigen is the pinned version") {
  // Eigen 5 keeps EIGEN_WORLD_VERSION at 3 for compatibility; the release is in the string.
  CHECK(std::string(EIGEN_VERSION_STRING) == "5.0.1");
}

TEST_CASE("dependencies: Spectra finds the extreme eigenvalues of a path-graph Laplacian") {
  // Oracle (closed form): the Laplacian of the path graph on n vertices has eigenvalues
  // 2 - 2 cos(k pi / n), k = 0..n-1. Spectra stops once the residual norm is below
  // tol * |theta|; for a symmetric matrix |theta - lambda| <= residual norm, and |lambda| <= 4,
  // so 4 * tol bounds the error. Rounding in the matrix products, n * eps * ||A|| with
  // ||A|| <= 4, is below 2e-13 for n = 200, far below that bound.
  const int n = 200;
  std::vector<Eigen::Triplet<double>> entries;
  for (int i = 0; i < n; ++i) {
    entries.emplace_back(i, i, (i == 0 || i == n - 1) ? 1.0 : 2.0);
    if (i + 1 < n) {
      entries.emplace_back(i, i + 1, -1.0);
      entries.emplace_back(i + 1, i, -1.0);
    }
  }
  Eigen::SparseMatrix<double> lap(n, n);
  lap.setFromTriplets(entries.begin(), entries.end());

  Spectra::SparseSymMatProd<double> op(lap);
  Spectra::SymEigsSolver<Spectra::SparseSymMatProd<double>> eigs(op, 4, 20);
  eigs.init();
  const double tol = 1e-10;
  const Eigen::Index nconv = eigs.compute(Spectra::SortRule::LargestAlge, 1000, tol);
  REQUIRE(eigs.info() == Spectra::CompInfo::Successful);
  REQUIRE(nconv == 4);
  const Eigen::VectorXd theta = eigs.eigenvalues();
  for (int j = 0; j < 4; ++j) {
    const double lambda = 2.0 - 2.0 * std::cos((n - 1 - j) * std::numbers::pi / n);
    CAPTURE(j);
    CHECK(std::abs(theta(j) - lambda) <= 4.0 * tol);
  }
}
