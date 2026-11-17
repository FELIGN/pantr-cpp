// _pantr.geometry and _pantr.transform: AABB and AffineTransform, for pantr.geometry's and
// pantr.transform's wrappers. Arguments arrive normalised (C-contiguous float64); the wrappers
// keep the checks whose messages spell a NumPy repr.
#include "arrays.hpp"
#include "interface.hpp"

#include "pantr/core/errors.hpp"
#include "pantr/geometry/aabb.hpp"
#include "pantr/transform/affine_transform.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>

#include <cstddef>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace pantr::bindings {

namespace {

using geometry::AABB;
using transform::AffineTransform;

using Vec = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using AnyArray = nb::ndarray<const double, nb::c_contig, nb::device::cpu>;

std::span<const double> span_of(const Vec& v) { return {v.data(), v.shape(0)}; }

std::span<const double> span_of(const AnyArray& a) { return {a.data(), a.size()}; }

std::vector<std::size_t> shape_of(const AnyArray& a) {
  std::vector<std::size_t> s(a.ndim());
  for (std::size_t i = 0; i < a.ndim(); ++i) {
    s[i] = a.shape(i);
  }
  return s;
}

std::optional<std::span<const double>> maybe(const std::optional<Vec>& v) {
  if (!v) {
    return std::nullopt;
  }
  return span_of(*v);
}

template <class R>
NumpyView<double> view_vector(std::span<const double> v, std::shared_ptr<const R> keep) {
  const std::size_t shape[1] = {v.size()};
  return view_of<double>(v.data(), shape, std::move(keep));
}

}  // namespace

void register_geometry(nb::module_& m) {
  nb::class_<AABB>(m, "AABB", "An immutable axis-aligned bounding box.")
      .def(
          "__init__",
          [](AABB* self, const Vec& lo, const Vec& hi) {
            new (self) AABB(span_of(lo), span_of(hi));
          },
          nb::arg("lo"), nb::arg("hi"), "Validate and copy the corners.")
      .def_static("unbounded", &AABB::unbounded, nb::arg("ndim"), "The box [-inf, inf]^ndim.")
      .def_static("empty", &AABB::empty, nb::arg("ndim"), "The canonical empty box.")
      .def_static(
          "from_bounds",
          [](const AnyArray& bounds) {
            const std::vector<std::size_t> shape = shape_of(bounds);
            return AABB::from_bounds(span_of(bounds), shape);
          },
          nb::arg("bounds"), "A box from an (ndim, 2) array of [lo, hi] rows.")
      .def_prop_ro("ndim", &AABB::ndim, "The number of axes.")
      .def_prop_ro(
          "lo", [](const AABB& b) { return view_vector(b.lo(), b.rep()); },
          nb::rv_policy::automatic, "Read-only lower corner.")
      .def_prop_ro(
          "hi", [](const AABB& b) { return view_vector(b.hi(), b.rep()); },
          nb::rv_policy::automatic, "Read-only upper corner.")
      .def("is_empty", &AABB::is_empty, "Whether lo > hi on some axis.")
      .def(
          "contains_point",
          [](const AABB& b, const Vec& x) { return b.contains_point(span_of(x)); }, nb::arg("x"),
          "Whether the closed box contains x.")
      .def("overlaps", &AABB::overlaps, nb::arg("other"), "Whether the closed boxes intersect.")
      .def("union_with", &AABB::union_with, nb::arg("other"),
           "The smallest box containing both (an empty operand is returned as it is).")
      .def("intersect", &AABB::intersect, nb::arg("other"), "The intersection, or None.")
      .def(
          "pad", [](const AABB& b, const Vec& r) { return b.pad(span_of(r)); }, nb::arg("r"),
          "Grow each axis by r[i] on both sides.")
      .def(
          "pad_scalar", [](const AABB& b, double r) { return b.pad(r); }, nb::arg("r"),
          "Grow every axis by r on both sides.")
      .def(
          "transform",
          [](const AABB& b, long long affine_dim, const AnyArray& matrix, const Vec& offset) {
            const std::vector<std::size_t> shape = shape_of(matrix);
            return b.transform(affine_dim, span_of(matrix), shape, span_of(offset));
          },
          nb::arg("affine_dim"), nb::arg("matrix"), nb::arg("offset"),
          "The bounding box of the image under x -> A x + b.")
      .def(
          "as_bounds", [](const AABB& b) { return to_numpy(b.as_bounds()); },
          "A fresh (ndim, 2) array of [lo, hi] rows.")
      .def("equals", &AABB::operator==, nb::arg("other"), "IEEE value equality of the corners.")
      .def("same_rep", &AABB::same_rep, nb::arg("other"), "Whether both share one representation.");
}

void register_transform(nb::module_& m) {
  nb::class_<AffineTransform>(m, "AffineTransform", "An immutable affine map x -> A x + b.")
      .def(
          "__init__",
          [](AffineTransform* self, const AnyArray& matrix,
             const std::optional<AnyArray>& translation) {
            const std::vector<std::size_t> ms = shape_of(matrix);
            if (translation) {
              const std::vector<std::size_t> ts = shape_of(*translation);
              new (self) AffineTransform(span_of(matrix), ms, span_of(*translation), ts);
            } else {
              new (self) AffineTransform(span_of(matrix), ms);
            }
          },
          nb::arg("matrix"), nb::arg("translation").none(), "Validate and copy A and b.")
      .def_static(
          "identity",
          [](long long n) {
            if (n < 0) {
              throw value_error("negative dimensions are not allowed");
            }
            return AffineTransform::identity(static_cast<std::size_t>(n));
          },
          nb::arg("n"), "The identity on R^n.")
      .def_static(
          "translation",
          [](const Vec& offset) { return AffineTransform::translation(span_of(offset)); },
          nb::arg("offset"), "x -> x + offset.")
      .def_static(
          "scaling",
          [](const Vec& factors, const std::optional<Vec>& center) {
            return AffineTransform::scaling(span_of(factors), maybe(center));
          },
          nb::arg("factors"), nb::arg("center").none(), "x -> diag(factors) x about a center.")
      .def_static(
          "rotation_2d",
          [](double angle, const std::optional<Vec>& center) {
            return AffineTransform::rotation_2d(angle, maybe(center));
          },
          nb::arg("angle"), nb::arg("center").none(), "A rotation of R^2 about a center.")
      .def_static(
          "rotation_3d",
          [](double angle, const Vec& axis, const std::optional<Vec>& center) {
            return AffineTransform::rotation_3d(angle, span_of(axis), maybe(center));
          },
          nb::arg("angle"), nb::arg("axis"), nb::arg("center").none(),
          "A rotation of R^3 about an axis through a center.")
      .def_static(
          "mirror",
          [](const Vec& normal, const std::optional<Vec>& center) {
            return AffineTransform::mirror(span_of(normal), maybe(center));
          },
          nb::arg("normal"), nb::arg("center").none(), "A reflection through a hyperplane.")
      .def_static("shear", &AffineTransform::shear, nb::arg("dim"), nb::arg("component"),
                  nb::arg("direction"), nb::arg("factor"), "x[component] += factor x[direction].")
      .def_prop_ro("dim", &AffineTransform::dim, "The dimension n.")
      .def_prop_ro(
          "matrix",
          [](const AffineTransform& t) {
            const std::size_t shape[2] = {t.dim(), t.dim()};
            return view_of<double>(t.matrix().data(), shape, t.rep());
          },
          nb::rv_policy::automatic, "Read-only (n, n) matrix A.")
      .def_prop_ro(
          "offset", [](const AffineTransform& t) { return view_vector(t.offset(), t.rep()); },
          nb::rv_policy::automatic, "Read-only (n,) translation b.")
      .def("inverse", &AffineTransform::inverse, "The inverse map.")
      .def("compose", &AffineTransform::compose, nb::arg("other"), "self o other.")
      .def(
          "apply",
          [](const AffineTransform& t, const AnyArray& points) {
            const std::size_t n = t.dim() == 0 ? 0 : points.size() / t.dim();
            NdArray<double> out = [&] {
              const ReleaseIfParallel release(n, t.ops_per_point().ops_per_item);
              return t.apply(span_of(points));
            }();
            return to_numpy(std::move(out));  // with the GIL: it builds a Python object
          },
          nb::arg("points"), "A p + b per point, (m, n).");
}

}  // namespace pantr::bindings
