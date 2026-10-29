// Diagnostic only: Eigen PartialPivLU solve against the identity, row-major dynamic, float and
// double, as include/pantr/change_basis.hpp instantiates it.
#include <Eigen/Core>
#ifndef PROBE_NO_LU
#include <Eigen/LU>
#endif
template <class T>
using RowMatrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
template <class T>
void inv(const T* a, T* out, Eigen::Index m) {
  Eigen::Map<const RowMatrix<T>> am(a, m, m);
#ifndef PROBE_NO_LU
  const Eigen::PartialPivLU<RowMatrix<T>> lu(am);
  Eigen::Map<RowMatrix<T>>(out, m, m) = lu.solve(RowMatrix<T>::Identity(m, m));
#else
  Eigen::Map<RowMatrix<T>>(out, m, m) = am * am.transpose();
#endif
}
template void inv<float>(const float*, float*, Eigen::Index);
template void inv<double>(const double*, double*, Eigen::Index);
