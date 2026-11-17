// A consumer of an installed pantr: uses the core headers and the bundled third-party headers, and
// exits non-zero on any wrong result.
#include <pantr/pantr.hpp>

#include <Spectra/MatOp/DenseSymMatProd.h>
#include <Spectra/SymEigsSolver.h>
#include <Eigen/Core>

#include <cmath>
#include <cstdio>
#include <numbers>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAILED: %s\n", what);
    ++failures;
  }
}

}  // namespace

int main() {
  check(pantr::version.major == PANTR_VERSION_MAJOR, "version");

  // Parallel loop with disjoint writes: out[i] = 2 i.
  std::vector<double> out(100000, -1.0);
  pantr::parallel::parallel_for(out.size(), std::size_t{1000}, [&](std::size_t b, std::size_t e) {
    for (std::size_t i = b; i < e; ++i) {
      out[i] = 2.0 * static_cast<double>(i);
    }
  });
  bool all = true;
  for (std::size_t i = 0; i < out.size(); ++i) {
    all = all && out[i] == 2.0 * static_cast<double>(i);
  }
  check(all, "parallel_for writes every item");

  // Error hierarchy and Python spelling.
  try {
    pantr::parallel::set_num_threads(0);
    check(false, "set_num_threads(0) throws");
  } catch (const pantr::value_error& e) {
    check(std::string(e.what()) == "n must be >= 1, got 0", "set_num_threads message");
  }
  check(pantr::py_repr(1e-5) == "1e-05", "py_repr");

  // Exact rationals: 1/3 + 1/6 == 1/2.
  const auto half = pantr::exact::make_rational(1, 3) + pantr::exact::make_rational(1, 6);
  check(half == pantr::exact::make_rational(1, 2), "exact rational sum");

  // NdArray through an mdspan view.
  pantr::NdArray<double> a({2, 3});
  auto v = a.view<2>();
  v(1, 2) = 5.0;
  check(a.data()[5] == 5.0, "NdArray row-major view");

  // Bundled Eigen + Spectra: largest eigenvalues of the path-graph Laplacian on n vertices are
  // 2 - 2 cos(k pi / n), k = n-1, n-2, n-3. Spectra stops when the residual is below
  // tol * |theta|, and for a symmetric matrix |theta - lambda| <= residual, so 4 * tol bounds the
  // error (|lambda| <= 4) up to rounding, which is far below it.
  const int n = 20;
  Eigen::MatrixXd lap = Eigen::MatrixXd::Zero(n, n);
  for (int i = 0; i < n; ++i) {
    lap(i, i) = (i == 0 || i == n - 1) ? 1.0 : 2.0;
    if (i + 1 < n) {
      lap(i, i + 1) = -1.0;
      lap(i + 1, i) = -1.0;
    }
  }
  Spectra::DenseSymMatProd<double> op(lap);
  Spectra::SymEigsSolver<Spectra::DenseSymMatProd<double>> eigs(op, 3, 8);
  eigs.init();
  const double tol = 1e-10;
  eigs.compute(Spectra::SortRule::LargestAlge, 1000, tol);
  check(eigs.info() == Spectra::CompInfo::Successful, "Spectra converged");
  const Eigen::VectorXd theta = eigs.eigenvalues();
  for (int j = 0; j < 3; ++j) {
    const double lambda = 2.0 - 2.0 * std::cos((n - 1 - j) * std::numbers::pi / n);
    check(std::abs(theta(j) - lambda) <= 4.0 * tol, "Spectra eigenvalue");
  }

  if (failures == 0) {
    std::printf("pantr consumer: all checks passed (pantr %d.%d.%d, simd %s)\n",
                pantr::version.major, pantr::version.minor, pantr::version.patch,
                pantr::simd::isa_name);
  }
  return failures == 0 ? 0 : 1;
}
