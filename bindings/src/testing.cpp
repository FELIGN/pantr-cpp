// _pantr._testing: probes for the tests and the benchmark harness. Not a public API.
#include "interface.hpp"
#include "probes.hpp"

#include "pantr/core/errors.hpp"
#include "pantr/parallel/shared.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace nb = nanobind;

namespace pantr::bindings {

namespace {

/// A class with one read-only integer property: the binding's attribute-read cost.
class ProbeObject {
 public:
  explicit ProbeObject(std::int64_t degree) : degree_(degree) {}
  [[nodiscard]] std::int64_t degree() const noexcept { return degree_; }

 private:
  std::int64_t degree_;
};

using PointsIn = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using TableOut = nb::ndarray<nb::numpy, double, nb::ndim<2>>;

/// Bernstein basis of `degree` at each point, by the de Casteljau triangle.
///
/// The measurement vehicle of the stage-0 stable-ABI check (one small array in, one out), not
/// pantr's Bernstein tabulation, which its own stage ports with its validation and oracles.
TableOut probe_bernstein(std::int64_t degree, const PointsIn& pts) {
  if (degree < 0) {
    throw pantr::value_error("degree must be >= 0, got " + std::to_string(degree));
  }
  const auto p = static_cast<std::size_t>(degree);
  const std::size_t n = pts.shape(0);
  constexpr std::size_t max_doubles = (std::numeric_limits<std::size_t>::max)() / sizeof(double);
  if (p >= max_doubles || (n != 0 && p + 1 > max_doubles / n)) {
    throw pantr::capacity_error("a table of " + std::to_string(n) + " points by degree " +
                                std::to_string(degree) + " does not fit in memory");
  }
  auto* data = new double[n * (p + 1)];
  nb::capsule owner(data, [](void* q) noexcept { delete[] static_cast<double*>(q); });
  for (std::size_t i = 0; i < n; ++i) {
    const double t = pts(i);
    double* row = data + i * (p + 1);
    row[0] = 1.0;
    for (std::size_t j = 1; j <= p; ++j) {
      double saved = 0.0;
      for (std::size_t k = 0; k < j; ++k) {
        const double value = row[k];
        row[k] = saved + (1.0 - t) * value;
        saved = t * value;
      }
      row[j] = saved;
    }
  }
  return TableOut(data, {n, p + 1}, owner);
}

}  // namespace

void register_testing(nb::module_& m) {
  m.def("count_threads_used", &count_threads_used, nb::arg("chunks"), nb::arg("hold_us"),
        nb::call_guard<nb::gil_scoped_release>(),
        "Count the distinct threads that run a parallel loop of `chunks` chunks holding "
        "`hold_us` microseconds each.");
  m.def("routes_shared_parallelism", &pantr::parallel::routes_shared_parallelism,
        "Report whether this module routes its parallel state through another module.");
  m.def(
      "probe_scalar", [](double x) { return 2.0 * x; }, nb::arg("x"),
      "Return 2 * x: the binding's cost for one float in, one float out.");
  m.def("probe_bernstein", &probe_bernstein, nb::arg("degree"), nb::arg("pts"),
        "Bernstein basis by de Casteljau: the binding's cost for one small array in and out.");
  nb::class_<ProbeObject>(m, "ProbeObject", "An object with one read-only integer property.")
      .def(nb::init<std::int64_t>(), nb::arg("degree"))
      .def_prop_ro("degree", &ProbeObject::degree, "The value given at construction.");
}

}  // namespace pantr::bindings
