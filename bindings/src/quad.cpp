// _pantr.quad: the 1D rules, the Lagrange nodes, PointsLattice and QuadratureRule, for
// pantr.quad's wrappers; and _pantr.tolerance: the preset factors.
#include "arrays.hpp"
#include "interface.hpp"

#include "pantr/basis/lagrange_variant.hpp"
#include "pantr/quad/points_lattice.hpp"
#include "pantr/quad/quadrature_rule.hpp"
#include "pantr/quad/rules_1d.hpp"
#include "pantr/tolerance.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace pantr::bindings {

namespace {

template <class T>
using RulePair = std::tuple<NumpyArray<T>, NumpyArray<T>>;

template <class T, quad::Rule1D<T> (*Rule)(long long)>
RulePair<T> rule(long long n_pts) {
  quad::Rule1D<T> r = Rule(n_pts);
  return {to_numpy(std::move(r.nodes)), to_numpy(std::move(r.weights))};
}

template <class T>
void def_rules(nb::module_& m, const std::string& suffix) {
  m.def(("trapezoidal_1d" + suffix).c_str(), &rule<T, &quad::trapezoidal_1d<T>>, nb::arg("n_pts"),
        "Trapezoidal rule on [0, 1]: (nodes, weights).");
  m.def(("gauss_legendre_1d" + suffix).c_str(), &rule<T, &quad::gauss_legendre_1d<T>>,
        nb::arg("n_pts"), "Gauss-Legendre rule on [0, 1]: (nodes, weights).");
  m.def(("gauss_lobatto_legendre_1d" + suffix).c_str(),
        &rule<T, &quad::gauss_lobatto_legendre_1d<T>>, nb::arg("n_pts"),
        "Gauss-Lobatto-Legendre rule on [0, 1]: (nodes, weights).");
  m.def(("chebyshev_gauss_1st_kind_1d" + suffix).c_str(),
        &rule<T, &quad::chebyshev_gauss_1st_kind_1d<T>>, nb::arg("n_pts"),
        "Chebyshev-Gauss rule of the first kind on [0, 1]: (nodes, weights).");
  m.def(("chebyshev_gauss_2nd_kind_1d" + suffix).c_str(),
        &rule<T, &quad::chebyshev_gauss_2nd_kind_1d<T>>, nb::arg("n_pts"),
        "Chebyshev-Gauss rule of the second kind on [0, 1]: (nodes, weights).");
  m.def(("tanh_sinh_1d" + suffix).c_str(), &rule<T, &quad::tanh_sinh_1d<T>>, nb::arg("n_pts"),
        "Tanh-sinh rule on [0, 1], truncated where the endpoint gap stops being resolvable.");
  m.def(("modified_chebyshev_nodes_1d" + suffix).c_str(),
        [](long long n_pts) { return to_numpy(quad::modified_chebyshev_nodes_1d<T>(n_pts)); },
        nb::arg("n_pts"), "Chebyshev-Lobatto nodes on [0, 1].");
  m.def(("lagrange_points" + suffix).c_str(),
        [](int variant, long long n_pts) {
          return to_numpy(basis::lagrange_points<T>(basis::to_lagrange_variant(variant), n_pts));
        },
        nb::arg("variant"), nb::arg("n_pts"), "Nodes of a Lagrange basis on [0, 1].");
}

template <class T>
using In1D = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <class T>
void def_lattice(nb::module_& m, const char* name, const std::string& suffix) {
  using Lattice = quad::PointsLattice<T>;
  nb::class_<Lattice>(m, name, "An immutable tensor-product lattice of points.")
      .def(
          "__init__",
          [](Lattice* self, const std::vector<In1D<T>>& pts) {
            std::vector<std::vector<T>> copy;
            copy.reserve(pts.size());
            for (const auto& a : pts) {
              copy.emplace_back(a.data(), a.data() + a.shape(0));
            }
            new (self) Lattice(std::move(copy));
          },
          nb::arg("pts_per_dir"), "Copy the coordinate arrays into a new lattice.")
      .def_prop_ro("dim", &Lattice::dim, "The number of directions.")
      .def_prop_ro("num_points", &Lattice::num_points, "The number of points.")
      .def(
          "pts",
          [](const Lattice& self, std::size_t d) {
            const std::span<const T> p = self.pts(d);
            const std::size_t shape[1] = {p.size()};
            return view_of<T>(p.data(), shape, self.rep());
          },
          nb::arg("d"), "Read-only view of direction `d`'s coordinates.")
      .def(
          "all_points",
          [](const Lattice& self, int order) {
            return to_numpy(self.all_points(order == 0 ? Order::C : Order::F));
          },
          nb::arg("order"), "All points, (num_points, dim), in `order` (0 C, 1 F).");
  m.def(("lagrange_points_lattice" + suffix).c_str(),
        [](int variant, const std::vector<long long>& n_pts) {
          return quad::lagrange_points_lattice<T>(basis::to_lagrange_variant(variant), n_pts);
        },
        nb::arg("variant"), nb::arg("n_pts_per_dir"),
        "Lattice of the Lagrange nodes per direction.");
}

using AnyDouble = nb::ndarray<const double, nb::c_contig, nb::device::cpu>;

quad::DoubleArrayRef ref(const AnyDouble& a, std::vector<std::size_t>& shape) {
  shape.resize(a.ndim());
  for (std::size_t i = 0; i < a.ndim(); ++i) {
    shape[i] = a.shape(i);
  }
  return {std::span<const double>(a.data(), a.size()), shape};
}

}  // namespace

void register_quad(nb::module_& m) {
  def_rules<double>(m, "");
  def_rules<float>(m, "_f32");
  def_lattice<double>(m, "PointsLatticeF64", "");
  def_lattice<float>(m, "PointsLatticeF32", "_f32");
  nb::class_<quad::QuadratureRule>(m, "QuadratureRule",
                                   "An immutable quadrature rule on the unit cube.")
      .def(
          "__init__",
          [](quad::QuadratureRule* self, const AnyDouble& points, const AnyDouble& weights) {
            std::vector<std::size_t> ps;
            std::vector<std::size_t> ws;
            new (self) quad::QuadratureRule(ref(points, ps), ref(weights, ws));
          },
          nb::arg("points"), nb::arg("weights"), "Validate and copy the points and weights.")
      .def_prop_ro("ndim", &quad::QuadratureRule::ndim, "The number of axes.")
      .def_prop_ro("num_points", &quad::QuadratureRule::num_points, "The number of points.")
      .def_prop_ro(
          "points",
          [](const quad::QuadratureRule& r) {
            return view_of<double>(r.points().data(), r.points().shape(), r.rep());
          },
          nb::rv_policy::automatic, "Read-only (num_points, ndim) points.")
      .def_prop_ro(
          "weights",
          [](const quad::QuadratureRule& r) {
            return view_of<double>(r.weights().data(), r.weights().shape(), r.rep());
          },
          nb::rv_policy::automatic, "Read-only (num_points,) weights.");
  m.def(
      "tensor_product_quadrature",
      [](const std::vector<std::pair<AnyDouble, AnyDouble>>& rules) {
        std::vector<std::vector<std::size_t>> shapes(2 * rules.size());
        std::vector<std::pair<quad::DoubleArrayRef, quad::DoubleArrayRef>> refs;
        refs.reserve(rules.size());
        for (std::size_t d = 0; d < rules.size(); ++d) {
          refs.emplace_back(ref(rules[d].first, shapes[2 * d]),
                            ref(rules[d].second, shapes[2 * d + 1]));
        }
        return quad::tensor_product_quadrature(refs);
      },
      nb::arg("rules"), "Tensor product of (nodes, weights) pairs on [0, 1].");
  m.def(
      "gauss_legendre_quadrature",
      [](long long ndim, const std::vector<long long>& npts) {
        return quad::gauss_legendre_quadrature(ndim, npts);
      },
      nb::arg("ndim"), nb::arg("npts"), "Tensor-product Gauss-Legendre rule on the unit cube.");
}

void register_tolerance(nb::module_& m) {
  m.attr("strict_eps_factor") = tolerance::strict_eps_factor;
  m.attr("default_eps_factor") = tolerance::default_eps_factor;
  m.attr("conservative_eps_factor") = tolerance::conservative_eps_factor;
}

}  // namespace pantr::bindings
