// _pantr.bspline: SpanwiseElementExtraction, the element operators of a tensor-product space onto
// the Bezier, Lagrange or cardinal basis, applied matrix-free to one cell or to a batch; and, in
// _testing, the Kronecker application on arbitrary factors, its scratch size and the structural
// Lagrange identity mask.
//
// This translation unit reaches Eigen (the Lagrange and cardinal operators fetch their change-of-
// basis matrices from the process tables, whose builders solve with Eigen's LU). A batch runs on
// the pool, so it releases the GIL when its work warrants it; a one-cell application does not.
// Arguments arrive as the wrapper normalised them (C-contiguous arrays of the exact dtype, a
// writable `out`); validation of values is the C++ API's, with the messages the Python suite
// matches. The views of the compact operators, index maps and masks are read-only arrays over the
// value's shared representation.
#include "arrays.hpp"
#include "interface.hpp"

#include "pantr/basis/lagrange_variant.hpp"
#include "pantr/bspline/detail/kron_apply.hpp"
#include "pantr/bspline/extraction.hpp"
#include "pantr/bspline/space.hpp"
#include "pantr/bspline/space_1d.hpp"
#include "pantr/bspline/spanwise_extraction.hpp"
#include "pantr/core/errors.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace pantr::bindings {

namespace {

template <class T>
using In1D = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
template <class T>
using In2D = nb::ndarray<const T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
template <class T>
using In3D = nb::ndarray<const T, nb::ndim<3>, nb::c_contig, nb::device::cpu>;
template <class T>
using Out1D = nb::ndarray<nb::numpy, T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
template <class T>
using Out2D = nb::ndarray<nb::numpy, T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
template <class T>
using Out3D = nb::ndarray<nb::numpy, T, nb::ndim<3>, nb::c_contig, nb::device::cpu>;
template <class T>
using Scratch1D = nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
template <class T>
using Scratch2D = nb::ndarray<T, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using Cells = nb::ndarray<const std::int64_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using OutBool1D = nb::ndarray<nb::numpy, bool, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <class T>
std::span<const T> as_span(const In1D<T>& a) {
  return {a.data(), a.shape(0)};
}

template <class T>
std::span<T> as_span(Out1D<T>& a) {
  return {a.data(), a.shape(0)};
}

template <class T>
mdview<const T, 2> as_view(const In2D<T>& a) {
  return mdview<const T, 2>(a.data(), a.shape(0), a.shape(1));
}

template <class T>
mdview<const T, 3> as_view(const In3D<T>& a) {
  return mdview<const T, 3>(a.data(), a.shape(0), a.shape(1), a.shape(2));
}

template <class T>
mdview<T, 2> as_view(Out2D<T>& a) {
  return mdview<T, 2>(a.data(), a.shape(0), a.shape(1));
}

template <class T>
mdview<T, 3> as_view(Out3D<T>& a) {
  return mdview<T, 3>(a.data(), a.shape(0), a.shape(1), a.shape(2));
}

mdview<const std::int64_t, 2> as_view(const Cells& a) {
  return mdview<const std::int64_t, 2>(a.data(), a.shape(0), a.shape(1));
}

/// A scratch given as a 1D array, or none.
template <class T>
std::optional<std::span<T>> maybe_span(std::optional<Scratch1D<T>>& a) {
  if (!a.has_value()) {
    return std::nullopt;
  }
  return std::span<T>(a->data(), a->shape(0));
}

/// A scratch given as a 2D array, or none.
template <class T>
std::optional<mdview<T, 2>> maybe_view(std::optional<Scratch2D<T>>& a) {
  if (!a.has_value()) {
    return std::nullopt;
  }
  return mdview<T, 2>(a->data(), a->shape(0), a->shape(1));
}

template <class T>
void def_spanwise(nb::module_& m, const char* name) {
  using E = bspline::SpanwiseElementExtraction<T>;
  using Space = bspline::BsplineSpace<T>;
  using bspline::ExtractionOp;

  // The one-cell methods: the cell is the interval index in each direction.
  const auto def_vector_ops = [](nb::class_<E>& cls) {
    cls.def(
           "apply",
           [](const E& self, const In1D<T>& v, const std::vector<std::size_t>& cell, Out1D<T>& out,
              std::optional<Scratch1D<T>> scratch) {
             self.apply(as_span(v), cell, as_span(out), maybe_span(scratch));
           },
           nb::arg("v"), nb::arg("cell"), nb::arg("out").noconvert(), nb::arg("scratch").none(),
           "Write M v for the element `cell` (an interval per direction) into `out`.")
        .def(
            "apply_transpose",
            [](const E& self, const In1D<T>& v, const std::vector<std::size_t>& cell, Out1D<T>& out,
               std::optional<Scratch1D<T>> scratch) {
              self.apply_transpose(as_span(v), cell, as_span(out), maybe_span(scratch));
            },
            nb::arg("v"), nb::arg("cell"), nb::arg("out").noconvert(), nb::arg("scratch").none(),
            "Write M^T v for the element `cell` into `out`.")
        .def(
            "apply_mt_k_m",
            [](const E& self, const In2D<T>& k, const std::vector<std::size_t>& cell, Out2D<T>& out,
               std::optional<Scratch1D<T>> scratch) {
              self.apply_mt_k_m(as_view(k), cell, as_view(out), maybe_span(scratch));
            },
            nb::arg("k"), nb::arg("cell"), nb::arg("out").noconvert(), nb::arg("scratch").none(),
            "Write M^T K M for the element `cell` into `out`.")
        .def(
            "apply_m_k_mt",
            [](const E& self, const In2D<T>& k, const std::vector<std::size_t>& cell, Out2D<T>& out,
               std::optional<Scratch1D<T>> scratch) {
              self.apply_m_k_mt(as_view(k), cell, as_view(out), maybe_span(scratch));
            },
            nb::arg("k"), nb::arg("cell"), nb::arg("out").noconvert(), nb::arg("scratch").none(),
            "Write M K M^T for the element `cell` into `out`.");
  };

  nb::class_<E> cls(m, name,
                    "An immutable set of element operators of a tensor-product space onto a "
                    "target basis.");
  cls.def(
         "__init__",
         [](E* self, const Space& space, int target, int variant) {
           new (self)
               E(space, bspline::to_extraction_target(target), basis::to_lagrange_variant(variant));
         },
         nb::arg("space"), nb::arg("target"), nb::arg("variant"),
         "Build the operators (target 0 Bezier, 1 Lagrange, 2 cardinal); refuses periodic "
         "directions.")
      .def_prop_ro("dim", &E::dim, "The number of directions.")
      .def_prop_ro(
          "num_intervals",
          [](const E& e) {
            const auto s = e.num_intervals();
            return std::vector<std::size_t>(s.begin(), s.end());
          },
          "The intervals per direction.")
      .def_prop_ro(
          "input_shape_per_dir",
          [](const E& e) {
            const auto s = e.input_shape_per_dir();
            return std::vector<std::size_t>(s.begin(), s.end());
          },
          "The columns of each direction's operator.")
      .def_prop_ro(
          "output_shape_per_dir",
          [](const E& e) {
            const auto s = e.output_shape_per_dir();
            return std::vector<std::size_t>(s.begin(), s.end());
          },
          "The rows of each direction's operator.")
      .def_prop_ro("num_identity_elements", &E::num_identity_elements,
                   "The elements whose operator is the identity in every direction.")
      .def(
          "compact_ops",
          [](const E& e, std::size_t k) {
            const NdArray<T>& a = e.compact_ops(k);
            return view_of<T>(a.data(), a.shape(), e.rep());
          },
          nb::arg("k"), "Read-only view of direction `k`'s non-identity operators.")
      .def(
          "idx_map",
          [](const E& e, std::size_t k) {
            const std::span<const std::int64_t> s = e.idx_map(k);
            const std::size_t shape[1] = {s.size()};
            return view_of<std::int64_t>(s.data(), shape, e.rep());
          },
          nb::arg("k"), "Read-only view of direction `k`'s row of each interval.")
      .def(
          "identity_mask",
          [](const E& e, std::size_t k) {
            const std::span<const bool> s = e.identity_mask(k);
            const std::size_t shape[1] = {s.size()};
            return view_of<bool>(s.data(), shape, e.rep());
          },
          nb::arg("k"), "Read-only view of which intervals of direction `k` are identity.")
      .def(
          "required_scratch",
          [](const E& e, int op) { return e.required_scratch(bspline::to_extraction_op(op)); },
          nb::arg("op"), "The scratch entries an application of one cell needs.")
      .def(
          "kron_operator",
          [](const E& self, const std::vector<std::size_t>& cell, Out2D<T>& out) {
            self.kron_operator(cell, as_view(out));
          },
          nb::arg("cell"), nb::arg("out").noconvert(),
          "Write the Kronecker operator of the element `cell` into `out`.")
      .def(
          "apply_many",
          [](const E& self, const In2D<T>& v, const Cells& cells, Out2D<T>& out,
             std::optional<Scratch2D<T>> scratch) {
            const ReleaseIfParallel gil(cells.shape(0), self.work_per_cell(ExtractionOp::apply));
            self.apply_many(as_view(v), as_view(cells), as_view(out), maybe_view(scratch));
          },
          nb::arg("v"), nb::arg("cells"), nb::arg("out").noconvert(), nb::arg("scratch").none(),
          "Write M_c v[c] for every cell (rows of `cells`) into `out`.")
      .def(
          "apply_transpose_many",
          [](const E& self, const In2D<T>& v, const Cells& cells, Out2D<T>& out,
             std::optional<Scratch2D<T>> scratch) {
            const ReleaseIfParallel gil(cells.shape(0),
                                        self.work_per_cell(ExtractionOp::apply_transpose));
            self.apply_transpose_many(as_view(v), as_view(cells), as_view(out),
                                      maybe_view(scratch));
          },
          nb::arg("v"), nb::arg("cells"), nb::arg("out").noconvert(), nb::arg("scratch").none(),
          "Write M_c^T v[c] for every cell into `out`.")
      .def(
          "apply_mt_k_m_many",
          [](const E& self, const In3D<T>& k, const Cells& cells, Out3D<T>& out,
             std::optional<Scratch2D<T>> scratch) {
            const ReleaseIfParallel gil(cells.shape(0), self.work_per_cell(ExtractionOp::mt_k_m));
            self.apply_mt_k_m_many(as_view(k), as_view(cells), as_view(out), maybe_view(scratch));
          },
          nb::arg("k"), nb::arg("cells"), nb::arg("out").noconvert(), nb::arg("scratch").none(),
          "Write M_c^T K[c] M_c for every cell into `out`.")
      .def(
          "apply_m_k_mt_many",
          [](const E& self, const In3D<T>& k, const Cells& cells, Out3D<T>& out,
             std::optional<Scratch2D<T>> scratch) {
            const ReleaseIfParallel gil(cells.shape(0), self.work_per_cell(ExtractionOp::m_k_mt));
            self.apply_m_k_mt_many(as_view(k), as_view(cells), as_view(out), maybe_view(scratch));
          },
          nb::arg("k"), nb::arg("cells"), nb::arg("out").noconvert(), nb::arg("scratch").none(),
          "Write M_c K[c] M_c^T for every cell into `out`.");
  def_vector_ops(cls);
}

/// The Kronecker application on arbitrary factors, for the tests: the factors, their identity
/// flags, and a flat operand and result.
template <class T>
void def_testing(nb::module_& m, const std::string& suffix) {
  m.def(("kron_apply" + suffix).c_str(),
        [](int op, const std::vector<In2D<T>>& factors, const std::vector<bool>& identity,
           const nb::ndarray<const T, nb::c_contig, nb::device::cpu>& operand,
           nb::ndarray<nb::numpy, T, nb::c_contig, nb::device::cpu>& out,
           const std::optional<Scratch1D<T>>& scratch) {
          if (identity.size() != factors.size()) {
            throw value_error("is_identity_per_dir has length " + std::to_string(identity.size()) +
                              ", expected " + std::to_string(factors.size()) +
                              " to match ops_1d_per_cell");
          }
          std::vector<bspline::KronFactor<T>> f;
          f.reserve(factors.size());
          for (std::size_t k = 0; k < factors.size(); ++k) {
            f.push_back({std::span<const T>(factors[k].data(), factors[k].size()),
                         factors[k].shape(0), factors[k].shape(1), identity[k]});
          }
          std::vector<std::size_t> rows;
          std::vector<std::size_t> cols;
          for (const auto& x : f) {
            rows.push_back(x.rows);
            cols.push_back(x.cols);
          }
          const bspline::ExtractionOp o = bspline::to_extraction_op(op);
          std::vector<T> own;
          std::span<T> work;
          if (scratch.has_value()) {
            work = std::span<T>(scratch->data(), scratch->shape(0));
          } else {
            own.resize(bspline::required_scratch_size(o, rows, cols));
            work = own;
          }
          bspline::apply_kron<T>(o, f, std::span<const T>(operand.data(), operand.size()),
                                 std::span<T>(out.data(), out.size()), work);
        },
        nb::arg("op"), nb::arg("factors"), nb::arg("identity"), nb::arg("operand"),
        nb::arg("out").noconvert(), nb::arg("scratch").none(),
        "Apply kron(factors) to a flat operand, as one cell of a SpanwiseElementExtraction does.");
  m.def(("lagrange_identity_mask" + suffix).c_str(),
        [](const bspline::BsplineSpace1D<T>& space, int variant, OutBool1D& out) {
          bspline::lagrange_structural_identity_mask<T>(space, basis::to_lagrange_variant(variant),
                                                        std::span<bool>(out.data(), out.shape(0)));
        },
        nb::arg("space"), nb::arg("variant"), nb::arg("out").noconvert(),
        "Write which intervals have an identity Lagrange extraction operator into `out`.");
}

}  // namespace

void register_bspline_spanwise(nb::module_& m) {
  def_spanwise<double>(m, "SpanwiseExtractionF64");
  def_spanwise<float>(m, "SpanwiseExtractionF32");
}

void register_bspline_spanwise_testing(nb::module_& m) {
  def_testing<double>(m, "");
  def_testing<float>(m, "_f32");
  m.def(
      "required_scratch_size",
      [](int op, const std::vector<std::size_t>& rows, const std::vector<std::size_t>& cols) {
        return bspline::required_scratch_size(bspline::to_extraction_op(op), rows, cols);
      },
      nb::arg("op"), nb::arg("rows"), nb::arg("cols"),
      "The scratch entries an operation on factors of these shapes needs per cell.");
}

}  // namespace pantr::bindings
