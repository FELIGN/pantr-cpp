// _pantr.mpi: the distributed algorithms of pantr::mpi, instantiated over AnyCommunicator so that
// they are compiled once, here, for every communicator a caller may pass (design §5.2). Two
// producers of an AnyCommunicator: a capsule from _pantr_mpi (an mpi4py communicator over real MPI,
// through the comm_capsule.h table), and PyCommunicator, which calls a duck-typed Python object's
// `rank`, `size`, `allgather` and `allreduce` with the payloads pantr.mpi has always sent (§5.4).
// No MPI here. The GIL is held throughout: PyCommunicator calls Python, and the collectives run on
// the calling thread. Reaches Eigen's LU and SVD (the collocation and mass solves).
#include "arrays.hpp"
#include "interface.hpp"

#include "pantr/bspline/fit.hpp"
#include "pantr/bspline/l2_project.hpp"
#include "pantr/grid/partition.hpp"
#include "pantr/mpi/collocation.hpp"
#include "pantr/mpi/comm_capsule.h"
#include "pantr/mpi/communicator.hpp"
#include "pantr/mpi/distributed_space.hpp"
#include "pantr/mpi/l2.hpp"
#include "pantr/mpi/partition_from_owner.hpp"
#include "pantr/mpi/quasi_interpolate.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace pantr::bindings {

namespace {

using mpi::AnyCommunicator;
using mpi::Block;
using mpi::Gathered;

template <class T>
using Any = nb::ndarray<const T, nb::c_contig, nb::device::cpu>;

template <class T>
using Vec = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <class T>
std::vector<std::size_t> shape_of(const Any<T>& a) {
  std::vector<std::size_t> s(a.ndim());
  for (std::size_t i = 0; i < s.size(); ++i) {
    s[i] = a.shape(i);
  }
  return s;
}

template <class T>
std::span<const T> flat_of(const Any<T>& a) {
  return {a.data(), a.size()};
}

template <class T>
const char* numpy_dtype() {
  if constexpr (std::is_same_v<T, std::int64_t>) {
    return "int64";
  } else if constexpr (std::is_same_v<T, double>) {
    return "float64";
  } else {
    return "float32";
  }
}

/// A fresh NumPy array holding a copy of a block, with its shape.
template <class T>
nb::object to_python(std::span<const T> data, const std::vector<std::size_t>& shape) {
  NdArray<T> a(std::vector<std::size_t>(shape), for_overwrite);
  std::copy(data.begin(), data.end(), a.data());
  return nb::cast(to_numpy(std::move(a)));
}

/// `int(x)`, as Python spells it.
int as_int(nb::handle x) { return nb::cast<int>(nb::module_::import_("builtins").attr("int")(x)); }

/// `numpy.asarray(x, dtype)` made C-contiguous, keeping its rank (a 0-d entry stays 0-d, as the
/// previous Python's `numpy.asarray` kept it, so a consumer can refuse it).
nb::object contiguous(const nb::module_& np, nb::handle x, const char* dtype) {
  return np.attr("require")(np.attr("asarray")(x, nb::arg("dtype") = dtype),
                            nb::arg("requirements") = "C");
}

/// Read one gathered entry as `contiguous(x, dtype)`, appending it to `out`.
template <class T>
void append_entry(const nb::module_& np, nb::handle x, Gathered<T>& out) {
  const auto a = nb::cast<Any<T>>(contiguous(np, x, numpy_dtype<T>()));
  out.shapes.push_back(shape_of(a));
  out.data.insert(out.data.end(), a.data(), a.data() + a.size());
}

/// A duck-typed Python communicator (design §5.4): `rank` and `size` read as attributes when
/// needed (a communicator without `rank` works where no rank is read), `allgather(obj)` returning
/// one entry per rank, `allreduce(arr)` with the default operation (a sum). Holds a strong
/// reference to the object; copies share it. Used with the GIL held.
class PyCommunicator {
 public:
  explicit PyCommunicator(nb::object comm) : comm_(std::move(comm)) {}

  // Read as the previous Python read them, `int(comm.rank)`: a float is truncated, `None` raises
  // TypeError.
  [[nodiscard]] int rank() const { return as_int(comm_.attr("rank")); }
  [[nodiscard]] int size() const { return as_int(comm_.attr("size")); }

  template <class T>
  [[nodiscard]] Gathered<T> allgatherv(const Block<T>& b) const {
    mpi::check_block(b);
    nb::object result = comm_.attr("allgather")(to_python(b.data, b.shape));
    const nb::module_ np = nb::module_::import_("numpy");
    Gathered<T> out;
    for (nb::handle item : result) {
      append_entry(np, item, out);
    }
    return out;
  }

  [[nodiscard]] std::pair<Gathered<std::int64_t>, Gathered<double>> allgather_owned(
      const Block<std::int64_t>& index, const Block<double>& value) const {
    mpi::check_block(index);
    mpi::check_block(value);
    nb::object result = comm_.attr("allgather")(
        nb::make_tuple(to_python(index.data, index.shape), to_python(value.data, value.shape)));
    const nb::module_ np = nb::module_::import_("numpy");
    std::pair<Gathered<std::int64_t>, Gathered<double>> out;
    for (nb::handle item : result) {
      // Any pair, as the previous Python's tuple unpacking took it.
      const nb::list pair = nb::list(nb::borrow(item));
      if (pair.size() != 2) {
        throw value_error("allgather of (dofs, coefficients) returned an entry of " +
                          std::to_string(pair.size()) + " items; expected 2.");
      }
      append_entry(np, pair[0], out.first);
      append_entry(np, pair[1], out.second);
    }
    return out;
  }

  template <class T>
  [[nodiscard]] NdArray<T> allreduce_sum(std::span<const T> data,
                                         std::span<const std::size_t> shape) const {
    nb::object result = comm_.attr("allreduce")(
        to_python(data, std::vector<std::size_t>(shape.begin(), shape.end())));
    const nb::module_ np = nb::module_::import_("numpy");
    const auto a = nb::cast<Any<T>>(contiguous(np, result, numpy_dtype<T>()));
    NdArray<T> out(shape_of(a), for_overwrite);
    std::copy(a.data(), a.data() + a.size(), out.data());
    return out;
  }

 private:
  nb::object comm_;
};

static_assert(mpi::Communicator<PyCommunicator>);

/// The communicator a Python argument stands for: a `pantr.mpi.comm_v1` capsule (from _pantr_mpi),
/// or any other object, taken as a duck-typed communicator.
AnyCommunicator to_communicator(nb::handle comm) {
  if (PyCapsule_CheckExact(comm.ptr()) != 0) {
    void* p = PyCapsule_GetPointer(comm.ptr(), PANTR_MPI_COMM_CAPSULE_NAME);
    if (p == nullptr) {
      throw nb::python_error();
    }
    // The capsule owns the table; the shared pointer keeps the capsule alive (released with the
    // GIL held: the communicator never outlives the binding call that made it).
    auto* keep = new nb::object(nb::borrow(comm));
    std::shared_ptr<const pantr_mpi_comm> table(static_cast<const pantr_mpi_comm*>(p),
                                                [keep](const pantr_mpi_comm*) { delete keep; });
    return AnyCommunicator::from_table(std::move(table));
  }
  return AnyCommunicator(PyCommunicator(nb::borrow(comm)));
}

using Derivs = std::vector<std::optional<std::pair<long long, long long>>>;

template <class T>
std::string tag() {
  return sizeof(T) == 4 ? "f32" : "f64";
}

template <class P, class T>
void def_collocate(nb::module_& m) {
  using Cm = std::conditional_t<(sizeof(P) > sizeof(T)), P, T>;
  m.def(("collocate_" + tag<P>() + "_" + tag<T>()).c_str(),
        [](nb::handle comm, const bspline::TensorCollocation<P>& system, const Any<double>& local,
           double tol) {
          const AnyCommunicator c = to_communicator(comm);
          const Block<double> b{flat_of(local), shape_of(local)};
          return to_numpy(mpi::collocate_values_distributed<T, Cm>(c, system, b, tol));
        },
        nb::arg("comm"), nb::arg("system"), nb::arg("local").noconvert(), nb::arg("tol"),
        "Gather every rank's value block and solve the collocation system (nodes P, space T).");
}

template <class T, class Cm>
void def_l2(nb::module_& m) {
  m.def(("l2_project_" + tag<T>() + "_" + tag<Cm>()).c_str(),
        [](nb::handle comm, const bspline::L2Projection<T>& proj,
           const Vec<std::int32_t>& cell_owner, const Any<Cm>& values, std::size_t n_components,
           const std::vector<Any<Cm>>& traces, double tol) {
          const AnyCommunicator c = to_communicator(comm);
          std::vector<std::span<const Cm>> spans;
          spans.reserve(traces.size());
          for (const Any<Cm>& t : traces) {
            spans.push_back(flat_of(t));
          }
          return to_numpy(mpi::l2_project_values_distributed(
              c, proj, std::span<const std::int32_t>(cell_owner.data(), cell_owner.shape(0)),
              flat_of(values), n_components, spans, tol));
        },
        nb::arg("comm"), nb::arg("projection"), nb::arg("cell_owner").noconvert(),
        nb::arg("values").noconvert(), nb::arg("n_components"), nb::arg("traces").noconvert(),
        nb::arg("tol"), "Sum the owned-cell loads over the ranks and solve (space T, values Cm).");
}

}  // namespace

void register_mpi(nb::module_& m) {
  m.attr("COMM_CAPSULE_NAME") = PANTR_MPI_COMM_CAPSULE_NAME;
  m.attr("COMM_ABI_VERSION") = PANTR_MPI_COMM_ABI_VERSION;

  m.def(
      "broadcast_partition",
      [](nb::handle comm, const nb::typed<nb::callable, grid::Partition()>& compute) {
        const AnyCommunicator c = to_communicator(comm);
        return mpi::broadcast_partition(c, [&]() { return nb::cast<grid::Partition>(compute()); });
      },
      nb::arg("comm"), nb::arg("compute"),
      "Rank 0's partition on every rank; `compute()` runs on rank 0 only (every rank at size 1).");

  m.def(
      "block_bounds",
      [](std::size_t n_total, std::size_t n_parts, std::size_t rank) {
        return mpi::block_bounds(n_total, n_parts, rank);
      },
      nb::arg("n_total"), nb::arg("n_parts"), nb::arg("rank"),
      "This rank's half-open block of a contiguous range split evenly.");

  def_collocate<double, double>(m);
  def_collocate<double, float>(m);
  def_collocate<float, double>(m);
  def_collocate<float, float>(m);

  def_l2<double, double>(m);
  def_l2<float, double>(m);
  def_l2<float, float>(m);

  m.def(
      "gather_owned",
      [](nb::handle comm, std::int64_t n_global, const Vec<std::int64_t>& local_to_global_dof,
         const Vec<bool>& owned_dof_mask, const Any<double>& coeffs) {
        const AnyCommunicator c = to_communicator(comm);
        if (coeffs.ndim() != 1 && coeffs.ndim() != 2) {
          throw value_error("gather_owned: coefficients must be 1-D or 2-D.");
        }
        const bool scalar = coeffs.ndim() == 1;
        const std::size_t n = local_to_global_dof.shape(0);
        std::size_t rank = 1;
        if (!scalar) {
          rank = coeffs.shape(1);
        }
        mpi::OwnedCoefficients mine{{}, {}, rank, scalar};
        if (n > 0) {
          std::vector<std::uint8_t> mask(owned_dof_mask.shape(0));
          for (std::size_t i = 0; i < mask.size(); ++i) {
            mask[i] = owned_dof_mask.data()[i] ? 1 : 0;
          }
          mine =
              mpi::owned_coefficients(std::span<const std::int64_t>(local_to_global_dof.data(), n),
                                      mask, flat_of(coeffs), rank, scalar);
        }
        mpi::GlobalCoefficients g = mpi::gather_owned_coefficients(c, n_global, mine);
        std::vector<std::size_t> shape{static_cast<std::size_t>(n_global)};
        if (!g.scalar) {
          shape.push_back(g.rank);
        }
        NdArray<double> out(std::move(shape), for_overwrite);
        std::copy(g.coeffs.begin(), g.coeffs.end(), out.data());
        return std::pair<NumpyArray<double>, bool>(to_numpy(std::move(out)), g.scalar);
      },
      nb::arg("comm"), nb::arg("n_global"), nb::arg("local_to_global_dof").noconvert(),
      nb::arg("owned_dof_mask").noconvert(), nb::arg("coeffs").noconvert(),
      "Gather every rank's owned (DOF, coefficient) pairs into the global field.");

  m.def(
      "partition_from_owner",
      [](nb::handle comm, const Vec<std::int64_t>& local_ids, std::int64_t n_cells) {
        const AnyCommunicator c = to_communicator(comm);
        return mpi::partition_from_owner(
            c, std::span<const std::int64_t>(local_ids.data(), local_ids.shape(0)), n_cells);
      },
      nb::arg("comm"), nb::arg("local_ids").noconvert(), nb::arg("n_cells"),
      "The partition whose owner of each cell is the rank that sent it.");
}

}  // namespace pantr::bindings
