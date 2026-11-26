// _pantr.bezier (continued): degree reduction and minimisation, and fitting, for pantr.bezier's
// wrapper; and the internals the tests reach through _pantr._testing. A translation unit of its own
// because it reaches Eigen's SVD (bezier.cpp reaches no Eigen).
#include "arrays.hpp"
#include "interface.hpp"

#include "pantr/bezier/bezier.hpp"
#include "pantr/bezier/fit.hpp"
#include "pantr/bezier/reduce.hpp"
#include "pantr/core/binomial.hpp"
#include "pantr/core/errors.hpp"
#include "pantr/core/ndarray.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace pantr::bindings {

namespace {

using bezier::Bezier;

template <class T>
using Any = nb::ndarray<const T, nb::c_contig, nb::device::cpu>;

template <class T>
using Vec = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <class T>
using Mat = nb::ndarray<const T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

template <class T>
std::vector<std::size_t> shape_of(const Any<T>& a) {
  std::vector<std::size_t> s(a.ndim());
  for (std::size_t i = 0; i < a.ndim(); ++i) {
    s[i] = a.shape(i);
  }
  return s;
}

/// A read-only view of a process-table operator, kept alive by the table's shared pointer.
NumpyView<double> operator_view(std::shared_ptr<const NdArray<double>> op) {
  const NdArray<double>* raw = op.get();
  return view_of<double>(raw->data(), raw->shape(), std::move(op));
}

// The reductions and fits release the GIL when their work warrants it (`ReleaseIfLong`): none
// touches a Python object (the arguments are already C++ values or buffers the call keeps alive,
// and the result is converted after the guard ends). Each estimate below runs before the call
// validates its arguments, so it reads only what an invalid argument still allows.

/// Exact-integer steps of `bernstein_gram_exact(n)`: each of its `(n + 1)^2` entries forms
/// `C(n, i)`, `C(n, j)` and `C(2n, i + j)` with `comb_exact`, which takes `min(k, m - k)` steps for
/// `C(m, k)`. Summed over the entries by grouping them, in loops of `O(n)` terms rather than over
/// the `(n + 1)^2` entries: `2 (n + 1) sum_i min(i, n - i)` for the first two, and
/// `sum_s (k_s + 1) k_s` with `k_s = min(s, 2n - s)` for the third, `k_s + 1` pairs having
/// `i + j = s`.
double gram_steps(long long n) {
  double single = 0.0;
  double pairs = 0.0;
  for (long long i = 0; i <= n; ++i) {
    single += static_cast<double>((std::min)(i, n - i));
  }
  for (long long s = 0; s <= 2 * n; ++s) {
    const auto k = static_cast<double>((std::min)(s, 2 * n - s));
    pairs += (k + 1.0) * k;
  }
  return 2.0 * static_cast<double>(n + 1) * single + pairs;
}

/// Work of building the interpolating reduction operator from degree `p` to `q = p - t`, for the
/// GIL decision only, or zero when the process table holds it. A lower bound: it counts only the
/// steps of the Gram matrices the build assembles (degree `p` unless `q == 1`, degree `q` when
/// `q >= 2`; `interpolating_reduction_operator`), each a multi-precision multiplication and exact
/// division by a machine integer, and leaves out the gcd that reduces each entry and the exact
/// projection and elimination that follow. Each step is charged 16 operations, an admitted
/// heuristic rather than a derivation: two calls into the multi-precision library, one of them a
/// hardware integer division. Never tuned on a particular machine.
double operator_build_ops(long long p, long long t) {
  if (bezier::detail::reduction_operator_cached(bezier::ReductionKind::interpolating, p, t)) {
    return 0.0;
  }
  const long long q = p - t;
  const double steps = (q != 1 ? gram_steps(p) : 0.0) + (q >= 2 ? gram_steps(q) : 0.0);
  return 16.0 * steps;
}

/// Work of `reduce_degree` and `degree_reduction_error`, for the GIL decision only: the first
/// reduced direction applies a `(p - t + 1, p + 1)` operator to every column of the net, so it
/// costs `b.size() * (p - t + 1)` multiply-adds; plus the largest operator build a direction needs
/// (`operator_build_ops`; the maximum rather than the sum, a bound whether or not directions of one
/// key share a build). Zero for decrements `detail::check_decrements` refuses (a count other than
/// the dimension, a negative one, or one above its direction's degree; none positive counts nothing
/// below), since the call then does no work. Only the private binding reaches those: the public
/// wrapper refuses them first.
template <class T>
double reduction_ops(const Bezier<T>& b, const std::vector<long long>& decrements) {
  if (decrements.size() != b.dim()) {
    return 0.0;
  }
  double ops = 0.0;
  double build = 0.0;
  for (std::size_t d = 0; d < b.dim(); ++d) {
    const auto p = static_cast<long long>(b.degree(d));
    const long long t = decrements[d];
    if (t < 0 || t > p) {
      return 0.0;
    }
    if (t == 0) {
      continue;
    }
    build = (std::max)(build, operator_build_ops(p, t));
    if (ops == 0.0) {
      ops = static_cast<double>(b.size()) * static_cast<double>(p - t + 1);
    }
  }
  return ops + build;
}

/// Work of `minimize_degree`, for the GIL decision only: its first trial applies a `(p, p + 1)`
/// operator to every column of the net, `b.size() * p` multiply-adds for the first direction of
/// degree `p >= 1`; plus the largest first-trial operator build a direction needs
/// (`operator_build_ops`). Zero for a tolerance that is not positive, which runs no trial, and
/// for a call `minimize_degree` refuses before any work: a tolerance that is not finite, or a
/// direction of degree above `bincoeff_max_n`.
template <class T>
double minimization_ops(const Bezier<T>& b, std::optional<double> tol) {
  if (tol.has_value() && !(*tol > 0.0 && std::isfinite(*tol))) {
    return 0.0;
  }
  double ops = 0.0;
  double build = 0.0;
  for (std::size_t d = 0; d < b.dim(); ++d) {
    const auto p = static_cast<long long>(b.degree(d));
    if (p > bincoeff_max_n) {
      return 0.0;
    }
    if (p < 1) {
      continue;
    }
    build = (std::max)(build, operator_build_ops(p, 1));
    if (ops == 0.0) {
      ops = static_cast<double>(b.size()) * static_cast<double>(p);
    }
  }
  return ops + build;
}

template <class T>
void def_reduction(nb::module_& m) {
  m.def(
      "reduce_degree",
      [](const Bezier<T>& b, const std::vector<long long>& decrements) {
        const ReleaseIfLong release(reduction_ops(b, decrements));
        return bezier::reduce_degree(b, decrements);
      },
      nb::arg("bezier"), nb::arg("decrements"), "Endpoint-interpolating L2 degree reduction.");
  m.def(
      "degree_reduction_error",
      [](const Bezier<T>& b, const std::vector<long long>& decrements) {
        const ReleaseIfLong release(reduction_ops(b, decrements));
        return bezier::degree_reduction_error(b, decrements);
      },
      nb::arg("bezier"), nb::arg("decrements"), "The exact L2 error of reduce_degree.");
  m.def(
      "minimize_degree",
      [](const Bezier<T>& b, std::optional<double> tol) {
        const ReleaseIfLong release(minimization_ops(b, tol));
        return bezier::minimize_degree(b, tol);
      },
      nb::arg("bezier"), nb::arg("tol").none(), "The lowest degree within a relative tolerance.");
}

/// Work of `fit_tensor`, for the GIL decision only: the first direction's contraction multiplies
/// every value by `p_0 + 1` coefficients (`p_0 = n_0 - 1` when the degrees are not given), and each
/// direction's pseudo-inverse is an SVD of an `(n_d, p_d + 1)` matrix, whose bidiagonalisation
/// alone updates the trailing columns once per column, `n_d (p_d + 1)^2` multiply-adds.
template <class T>
double fit_tensor_ops(const Any<T>& values, const std::vector<Vec<T>>& nodes,
                      const std::optional<std::vector<long long>>& degrees) {
  double first = 0.0;
  if (degrees && !degrees->empty()) {
    first = static_cast<double>((std::max)((*degrees)[0], 0LL));
  } else if (!nodes.empty() && nodes[0].shape(0) > 0) {
    first = static_cast<double>(nodes[0].shape(0) - 1);
  }
  double svd = 0.0;
  for (std::size_t d = 0; d < nodes.size(); ++d) {
    const auto n = static_cast<double>(nodes[d].shape(0));
    double cols = n;
    if (degrees && d < degrees->size()) {
      cols = static_cast<double>((std::max)((*degrees)[d], 0LL)) + 1.0;
    }
    svd += n * cols * cols;
  }
  return static_cast<double>(values.size()) * (first + 1.0) + svd;
}

/// Work of `fit_scattered`, for the GIL decision only: the SVD of the `(n_pts, n_coef)` Vandermonde
/// matrix, whose bidiagonalisation alone updates the trailing columns once per column,
/// `n_pts n_coef^2` multiply-adds.
template <class T>
double fit_scattered_ops(const Mat<T>& pts, const std::vector<long long>& degrees) {
  double coefs = 1.0;
  for (const long long p : degrees) {
    coefs *= static_cast<double>((std::max)(p, 0LL)) + 1.0;
  }
  return static_cast<double>(pts.shape(0)) * coefs * coefs;
}

template <class T>
void def_fit(nb::module_& m, const std::string& sfx) {
  m.def(("fit_tensor" + sfx).c_str(),
        [](const Any<T>& values, const std::vector<Vec<T>>& nodes,
           std::optional<std::vector<long long>> degrees, std::optional<double> tol) {
          std::vector<std::span<const T>> spans;
          spans.reserve(nodes.size());
          for (const Vec<T>& n : nodes) {
            spans.emplace_back(n.data(), n.shape(0));
          }
          const std::vector<std::size_t> shape = shape_of(values);
          const ReleaseIfLong release(fit_tensor_ops(values, nodes, degrees));
          return bezier::fit_tensor<T>(std::span<const T>(values.data(), values.size()), shape,
                                       spans, degrees, tol);
        },
        nb::arg("values"), nb::arg("nodes"), nb::arg("degrees").none(), nb::arg("tol").none(),
        "Fit on a tensor grid: values (*n_pts, rank), one node array per direction.");
  m.def(("fit_scattered" + sfx).c_str(),
        [](const Mat<T>& values, const Mat<T>& pts, const std::vector<long long>& degrees,
           std::optional<double> tol) {
          const ReleaseIfLong release(fit_scattered_ops(pts, degrees));
          return bezier::fit_scattered<T>(
              mdview<const T, 2>(values.data(), values.shape(0), values.shape(1)),
              mdview<const T, 2>(pts.data(), pts.shape(0), pts.shape(1)), degrees, tol);
        },
        nb::arg("values"), nb::arg("pts"), nb::arg("degrees"), nb::arg("tol").none(),
        "Fit at scattered points: values (n_pts, rank), points (n_pts, dim).");
}

}  // namespace

void register_bezier_approx(nb::module_& m) {
  def_reduction<double>(m);
  def_reduction<float>(m);
  def_fit<double>(m, "");
  def_fit<float>(m, "_f32");
}

void register_bezier_approx_testing(nb::module_& m) {
  m.def(
      "interpolating_reduction_operator",
      [](long long degree, long long decrement) {
        return operator_view(bezier::cached_reduction_operator(bezier::ReductionKind::interpolating,
                                                               degree, decrement));
      },
      nb::arg("degree"), nb::arg("decrement"),
      "The endpoint-interpolating reduction operator (read-only, from the process table).");
  m.def(
      "l2_reduction_operator",
      [](long long degree, long long decrement) {
        return operator_view(
            bezier::cached_reduction_operator(bezier::ReductionKind::l2, degree, decrement));
      },
      nb::arg("degree"), nb::arg("decrement"));
  m.def(
      "elevation_matrix",
      [](long long degree, long long increment) {
        return to_numpy(bezier::elevation_matrix(degree, increment));
      },
      nb::arg("degree"), nb::arg("increment"));
  m.attr("reduction_cache_max_degree") = bezier::reduction_cache_max_degree;
  m.def("reduction_cache_size", &bezier::reduction_cache_size);
  m.def("reduction_cache_clear", &bezier::reduction_cache_clear);
  m.def(
      "squared_l2_norm",
      [](const Any<double>& c) {
        std::vector<std::size_t> shape = shape_of(c);
        if (shape.empty()) {
          shape.push_back(1);
        }
        return bezier::detail::squared_l2_norm(std::span<const double>(c.data(), c.size()), shape);
      },
      nb::arg("coeffs"));
  m.def("projected_quadrature_size", &bezier::detail::projected_quadrature_size, nb::arg("degree"));
  m.def(
      "tensor_gauss_weights",
      [](const std::vector<std::size_t>& num_nodes) {
        return to_numpy(bezier::detail::tensor_gauss_weights(num_nodes));
      },
      nb::arg("num_nodes"));
  m.def(
      "sample_projected",
      [](const Any<double>& ctrl,
         const std::vector<std::size_t>& num_nodes) -> std::optional<NumpyArray<double>> {
        const std::vector<std::size_t> shape = shape_of(ctrl);
        if (shape.size() < 2 || num_nodes.size() != shape.size() - 1) {
          throw value_error("sample_projected needs (*orders, rank + 1) and one count per order");
        }
        auto r = bezier::detail::sample_projected<double>(
            std::span<const double>(ctrl.data(), ctrl.size()), shape, num_nodes);
        if (!r) {
          return std::nullopt;
        }
        return to_numpy(std::move(*r));
      },
      nb::arg("ctrl"), nb::arg("num_nodes"));
  m.def(
      "projected_relative_deviation",
      [](std::optional<Any<double>> values, std::optional<Any<double>> trial,
         const Any<double>& w) {
        const auto to_opt = [](const std::optional<Any<double>>& a) {
          std::optional<NdArray<double>> o;
          if (a) {
            o.emplace(shape_of(*a), for_overwrite);
            std::copy(a->data(), a->data() + a->size(), o->data());
          }
          return o;
        };
        const std::size_t need = w.size() * (values ? values->shape(values->ndim() - 1) : 0);
        if ((values && values->size() != need) || (trial && trial->size() != need)) {
          throw value_error("projected_relative_deviation: (*num_nodes, rank) values expected");
        }
        NdArray<double> wa(shape_of(w), for_overwrite);
        std::copy(w.data(), w.data() + w.size(), wa.data());
        return bezier::detail::projected_relative_deviation(to_opt(values), to_opt(trial), wa);
      },
      nb::arg("values").none(), nb::arg("trial_values").none(), nb::arg("quad_weights"));
}

}  // namespace pantr::bindings
