// _pantr.roots: the Builtin root provider, for pantr.bezier's root-finding wrapper; and the
// provider's internals the tests reach through _pantr._testing. Arguments arrive normalised
// (C-contiguous, the exact dtype); results cross as NumPy arrays without a copy.
#include "arrays.hpp"
#include "interface.hpp"

#include "pantr/core/errors.hpp"
#include "pantr/core/ndarray.hpp"
#include "pantr/parallel/work_model.hpp"
#include "pantr/roots/roots.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/tuple.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace pantr::bindings {

namespace {

namespace rd = pantr::roots::detail;
using pantr::roots::Builtin;
using pantr::roots::FindRootsOptions;
using pantr::roots::RootSet;

template <class T>
using Vec = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <class T>
using Mat = nb::ndarray<const T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

template <class T>
std::span<const T> span_of(const Vec<T>& a) {
  return {a.data(), a.shape(0)};
}

template <class T>
mdview<const T, 2> view_of(const Mat<T>& a) {
  return mdview<const T, 2>(a.data(), a.shape(0), a.shape(1));
}

NumpyArray<double> vector_to_numpy(const std::vector<double>& v) {
  NdArray<double> a({v.size()}, for_overwrite);
  std::copy(v.begin(), v.end(), a.data());
  return to_numpy(std::move(a));
}

/// Whether a batch of `n` polynomials of `n_coeff` coefficients runs on the pool (then the GIL is
/// released around it).
bool batch_parallel(std::size_t n, std::size_t n_coeff) {
  return parallel::worth_parallel(
      n, parallel::WorkEstimate{rd::ops_per_polynomial(n_coeff > 0 ? n_coeff - 1 : 0)});
}

template <class T>
void def_provider(nb::module_& m, const std::string& sfx) {
  m.def(("find_roots" + sfx).c_str(),
        [](const Vec<T>& c, double param_tol, double value_tol) {
          const RootSet<double> r = Builtin{}.find_roots<T>(span_of(c), {param_tol, value_tol});
          NdArray<double> x({r.roots.size()}, for_overwrite);
          for (std::size_t i = 0; i < r.roots.size(); ++i) {
            x.data()[i] = r.roots[i].x;
          }
          return std::make_tuple(to_numpy(std::move(x)), static_cast<int>(r.completeness));
        },
        nb::arg("coeff"), nb::arg("param_tol"), nb::arg("value_tol"),
        "Roots of one Bernstein polynomial: (roots, completeness).");
  m.def(("find_roots_detailed" + sfx).c_str(),
        [](const Vec<T>& c, double param_tol, double value_tol) {
          const RootSet<double> r = Builtin{}.find_roots<T>(span_of(c), {param_tol, value_tol});
          const std::size_t k = r.roots.size();
          NdArray<double> x({k}, for_overwrite);
          NdArray<std::int32_t> claim({k}, for_overwrite);
          NdArray<std::int32_t> mult({k}, for_overwrite);
          for (std::size_t i = 0; i < k; ++i) {
            x.data()[i] = r.roots[i].x;
            claim.data()[i] = static_cast<std::int32_t>(r.roots[i].claim);
            mult.data()[i] = r.roots[i].multiplicity;
          }
          NdArray<double> gaps({r.gaps.size(), 2}, for_overwrite);
          for (std::size_t i = 0; i < r.gaps.size(); ++i) {
            gaps.data()[2 * i] = r.gaps[i].first;
            gaps.data()[2 * i + 1] = r.gaps[i].second;
          }
          return std::make_tuple(to_numpy(std::move(x)), to_numpy(std::move(claim)),
                                 to_numpy(std::move(mult)), static_cast<int>(r.completeness),
                                 to_numpy(std::move(gaps)));
        },
        nb::arg("coeff"), nb::arg("param_tol"), nb::arg("value_tol"),
        "Roots with claims: (roots, claims, multiplicities, completeness, gaps (k, 2)).");
  m.def(("find_roots_batch" + sfx).c_str(),
        [](const Mat<T>& c, double param_tol, double value_tol) {
          pantr::roots::BatchRootSet r = [&] {
            std::optional<nb::gil_scoped_release> release;
            if (batch_parallel(c.shape(0), c.shape(1))) {
              release.emplace();
            }
            return Builtin{}.find_roots_batch<T>(view_of(c), {param_tol, value_tol});
          }();
          NdArray<std::int64_t> counts({r.counts.size()}, for_overwrite);
          std::copy(r.counts.begin(), r.counts.end(), counts.data());
          NdArray<std::int32_t> comp({r.completeness.size()}, for_overwrite);
          for (std::size_t i = 0; i < r.completeness.size(); ++i) {
            comp.data()[i] = static_cast<std::int32_t>(r.completeness[i]);
          }
          return std::make_tuple(to_numpy(std::move(r.roots)), to_numpy(std::move(counts)),
                                 to_numpy(std::move(comp)));
        },
        nb::arg("coeffs"), nb::arg("param_tol"), nb::arg("value_tol"),
        "Roots of each row: (roots (n, max(degree, 1)) NaN-padded, counts, completeness).");
  m.def(("find_monotone_root" + sfx).c_str(),
        [](const Vec<T>& c, double param_tol) {
          return Builtin{}.find_monotone_root<T>(span_of(c), param_tol);
        },
        nb::arg("coeff"), nb::arg("param_tol"), "The root of a monotone polynomial, or NaN.");
  m.def(("find_monotone_root_batch" + sfx).c_str(),
        [](const Mat<T>& c, double param_tol) {
          std::vector<double> r = [&] {
            std::optional<nb::gil_scoped_release> release;
            if (batch_parallel(c.shape(0), c.shape(1))) {
              release.emplace();
            }
            return Builtin{}.find_monotone_roots_batch<T>(view_of(c), param_tol);
          }();
          return vector_to_numpy(r);
        },
        nb::arg("coeffs"), nb::arg("param_tol"), "The monotone root of each row, NaN for none.");
}

// The provider's internals, with the call conventions of the Numba kernels they replace (a
// root array and a count), for the tests that pin them. Both dtypes under one name: nanobind
// tries the float64 overload without conversion first.
template <class T>
void def_internals(nb::module_& m) {
  m.def(
      "de_casteljau_eval_scalar",
      [](const Vec<T>& c, double t) { return rd::eval<T>(span_of(c), t); }, nb::arg("coeff"),
      nb::arg("t"));
  m.def(
      "de_casteljau_eval_and_deriv_scalar",
      [](const Vec<T>& c, double t) {
        const auto [f, df] = rd::eval_and_deriv<T>(span_of(c), t);
        return std::make_tuple(f, df);
      },
      nb::arg("coeff"), nb::arg("t"));
  m.def(
      "subdivide_scalar",
      [](const Vec<T>& c, double a, double b) {
        return vector_to_numpy(rd::subdivide<T>(span_of(c), a, b));
      },
      nb::arg("coeff"), nb::arg("t_min"), nb::arg("t_max"));
  m.def(
      "newton_polish_scalar",
      [](const Vec<T>& c, double mid, double lo, double hi, double tol) {
        const rd::Polished p = rd::newton_polish<T>(span_of(c), mid, lo, hi, tol);
        return std::make_tuple(p.t, p.f, p.df);
      },
      nb::arg("coeff"), nb::arg("mid"), nb::arg("lo"), nb::arg("hi"), nb::arg("param_tol"));
  m.def(
      "solve_monotone_root_kernel",
      [](const Vec<T>& c, double tol, int max_iter) {
        return rd::solve_monotone<T>(span_of(c), tol, max_iter);
      },
      nb::arg("coeff"), nb::arg("param_tol"), nb::arg("max_iter") = rd::max_newton_iter);
  m.def(
      "yuksel_roots",
      [](const Vec<T>& c, double tol) {
        const std::vector<double> r = rd::yuksel_roots<T>(span_of(c), tol);
        return std::make_tuple(vector_to_numpy(r), static_cast<long long>(r.size()));
      },
      nb::arg("coeff"), nb::arg("tol"));
  m.def(
      "clip_roots_core",
      [](const Vec<T>& c, double param_tol, double geom_tol, int max_depth) {
        const rd::ClipResult r = rd::clip_roots<T>(span_of(c), param_tol, geom_tol, max_depth);
        return std::make_tuple(vector_to_numpy(r.candidates),
                               static_cast<long long>(r.candidates.size()));
      },
      nb::arg("root_coeff"), nb::arg("param_tol"), nb::arg("geom_tol"),
      nb::arg("max_depth") = rd::clip_max_depth);
  m.def(
      "dedup_roots",
      [](const Vec<double>& raw, long long n_roots, const Vec<T>& c, double param_tol,
         double geom_tol) {
        if (n_roots < 0 || static_cast<std::size_t>(n_roots) > raw.shape(0)) {
          throw value_error("n_roots must be in [0, len(raw_roots)]");
        }
        std::vector<double> v(raw.data(), raw.data() + n_roots);
        return vector_to_numpy(rd::dedup_roots<T>(std::move(v), span_of(c), param_tol, geom_tol));
      },
      nb::arg("raw_roots"), nb::arg("n_roots"), nb::arg("coeff"), nb::arg("param_tol"),
      nb::arg("geom_tol"));
}

}  // namespace

void register_roots(nb::module_& m) {
  m.attr("CLAIM_NONE") = static_cast<int>(pantr::roots::Claim::none);
  m.attr("CLAIM_COMPUTED") = static_cast<int>(pantr::roots::Claim::computed);
  m.attr("CLAIM_CERTIFIED") = static_cast<int>(pantr::roots::Claim::certified);
  m.attr("COMPLETE") = static_cast<int>(pantr::roots::Completeness::complete);
  m.attr("GAP_LOCALIZED") = static_cast<int>(pantr::roots::Completeness::gap_localized);
  m.attr("GAP_UNLOCALIZED") = static_cast<int>(pantr::roots::Completeness::gap_unlocalized);
  m.def(
      "capabilities",
      [] {
        const pantr::roots::Capabilities c = Builtin{}.capabilities();
        return std::make_tuple(static_cast<int>(c.max_claim), c.batch, c.monotone, c.multiplicity);
      },
      "The Builtin provider's capabilities: (max_claim, batch, monotone, multiplicity).");
  def_provider<double>(m, "");
  def_provider<float>(m, "_f32");
}

void register_roots_testing(nb::module_& m) {
  m.def("have_same_sign", &rd::have_same_sign, nb::arg("a"), nb::arg("b"));
  m.def("have_opposite_signs", &rd::have_opposite_signs, nb::arg("a"), nb::arg("b"));
  m.def("spans_zero", &rd::spans_zero, nb::arg("a"), nb::arg("b"));
  m.def(
      "count_sign_changes", [](const Vec<double>& c) { return rd::count_sign_changes(span_of(c)); },
      nb::arg("coeff"));
  m.def(
      "clip_hull_to_zero",
      [](const Vec<double>& c) {
        const rd::HullClip h = rd::clip_hull_to_zero(span_of(c));
        return std::make_tuple(h.t_lo, h.t_hi, h.found);
      },
      nb::arg("coeff"));
  m.attr("ROOTS_MAX_NEWTON_ITER") = rd::max_newton_iter;
  m.attr("ROOTS_CLIP_MAX_DEPTH") = rd::clip_max_depth;
  def_internals<double>(m);
  def_internals<float>(m);
}

}  // namespace pantr::bindings
