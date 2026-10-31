"""Tensor-product change-of-basis extraction across B-spline elements.

This module exposes :class:`SpanwiseElementExtraction`, a lazy tensor-product
change-of-basis object. The value lives in the C++ library
(``pantr::bspline::SpanwiseElementExtraction``): it builds the per-direction 1D extraction
operators once at construction time and applies the d-dimensional operator of an element, or of a
batch of elements, as one mode-wise contraction per direction.

Three targets are supported (the source basis is always the B-spline basis):

- ``"bezier"``:   Bernstein (Bézier) basis on each element.
- ``"lagrange"``: Lagrange basis on each element, at the chosen point
  distribution (see :class:`pantr.basis.LagrangeVariant`).
- ``"cardinal"``: cardinal B-spline basis on each element.

Identity short-circuit is used wherever possible. All three targets use
structural (multiplicity-based) identity predicates:

- ``"bezier"``: element ``e`` is identity iff both its boundary unique knots
  have multiplicity ``>= degree + 1``, i.e. the element is already a Bézier
  patch. Knot multiplicities are computed using ``space.tolerance``.
- ``"lagrange"``: for ``degree == 0`` every element is trivially identity.
  For ``degree > 0`` an element is identity iff its Bézier extraction is
  identity and the Lagrange-to-Bernstein matrix equals ``I`` (which holds when
  the Lagrange nodes coincide with the Bernstein abscissae, e.g. ``degree == 1``
  with equispaced, GLL, or Chebyshev-2nd nodes).
- ``"cardinal"``: structural mask from
  :meth:`BsplineSpace1D.get_cardinal_intervals` labels uniform-span intervals,
  on which the cardinal extraction operator is exactly the identity.
"""

from __future__ import annotations

import copy as _copy
import functools
import math
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, cast, get_args

import numpy as np
import numpy.typing as npt

from .._ext import ext
from ..basis import LagrangeVariant
from ..basis._basis_tabulate import _variant_code
from ..basis._basis_utils import _allocate_or_validate_out

if TYPE_CHECKING:
    from ._bspline_space_nd import BsplineSpace

_cpp: Any = ext.bspline
"""The compiled spanwise extractions (private; untyped: one class per dtype)."""

_F32: np.dtype[np.float32] = np.dtype(np.float32)

OpKind = Literal["apply", "apply_T", "MT_K_M", "M_K_MT"]
"""Tag distinguishing the four apply variants.

- ``"apply"``:    ``out = M @ v``           (input shape prod -> output shape prod)
- ``"apply_T"``:  ``out = M^T @ v``         (output shape prod -> input shape prod)
- ``"MT_K_M"``:   ``out = M^T @ K @ M``     (output x output -> input x input)
- ``"M_K_MT"``:   ``out = M @ K @ M^T``     (input x input -> output x output)

with ``M = kron(M_0, M_1, ..., M_{d-1})``.
"""

_OP_CODES: dict[str, int] = {"apply": 0, "apply_T": 1, "MT_K_M": 2, "M_K_MT": 3}
"""The compiled library's wire value of each :data:`OpKind`."""

_TARGET_CODES: dict[str, int] = {"bezier": 0, "lagrange": 1, "cardinal": 2}
"""The compiled library's wire value of each target."""

_ONE_CELL_METHODS: dict[str, str] = {
    "apply": "apply",
    "apply_T": "apply_transpose",
    "MT_K_M": "apply_mt_k_m",
    "M_K_MT": "apply_m_k_mt",
}
_BATCH_METHODS: dict[str, str] = {
    "apply": "apply_many",
    "apply_T": "apply_transpose_many",
    "MT_K_M": "apply_mt_k_m_many",
    "M_K_MT": "apply_m_k_mt_many",
}

_VALUE_ATTRS: frozenset[str] = frozenset(
    {
        "_space",
        "_target",
        "_lagrange_variant",
        "_h",
        "_dtype",
        "_compact_ops_1d",
        "_idx_maps_1d",
        "_is_identity_mask_1d",
        "_input_shape_per_dir",
        "_output_shape_per_dir",
        "ops_1d",
        "num_identity_elements",
    }
)
"""The attributes of an extraction that hold its value or are cached from it."""


Target = Literal["bezier", "lagrange", "cardinal"]
"""Supported target bases for spanwise element extraction."""


CellIndex = int | tuple[int, ...] | list[int] | npt.NDArray[np.int_]
"""Accepted cell-index forms: flat ``int``, or a per-direction integer sequence."""

CellIndicesBatch = npt.NDArray[np.int_] | list[int] | list[tuple[int, ...]] | list[list[int]]
"""Accepted batch cell-index forms.

May be:

- 1-D integer array or list of ``n_cells`` flat indices (row-major over
  :attr:`~SpanwiseElementExtraction.num_intervals`).
- 2-D integer array of shape ``(n_cells, d)`` with per-direction indices.
- List of per-direction integer tuples or lists of length ``d``.
"""


class SpanwiseElementExtraction:
    """Tensor-product change-of-basis operator across B-spline elements.

    For a :class:`BsplineSpace` of dimension ``d`` and a chosen ``target``
    basis, this class eagerly builds per-direction compact operator storage:
    only the non-identity rows of each direction's extraction operator array
    are retained, reducing memory for identity-heavy spaces (e.g. cardinal
    spaces on uniform meshes). Per-element d-dimensional operators are never
    materialized unless explicitly requested via :meth:`operator` or
    :meth:`tabulate`: instead the apply-style methods contract the operator
    one direction at a time, in the compiled library.

    With the current 1D builders all per-direction operators are square of
    size ``(degree_k + 1, degree_k + 1)``. The class also supports non-square
    per-direction operators, so new 1D builders can plug in without changes,
    provided such an operator is not flagged as the identity.

    The value is immutable: the operators are built at construction and are
    read-only, and the arrays this class returns are read-only views of them.
    The per-direction data is exposed as two complementary representations:

    - *Compact* (:attr:`compact_ops_1d`, :attr:`idx_maps_1d`,
      :attr:`is_identity_mask_1d`): primary storage.
    - *Dense* (:attr:`ops_1d`): the full ``(n_elements_k, n_out_k, n_in_k)``
      layout, reconstructed lazily from compact storage on first access.

    For one element at a time, :meth:`factors` returns the per-direction
    ``(is_identity, operator)`` pairs without allocating anything, which is what
    a consumer serializing the 1D factors should use; ``__getitem__`` is the
    convenience form that materializes an explicit identity matrix instead.

    Any number of directions is supported.

    Attributes:
        _h (Any): The compiled value (``SpanwiseExtractionF64`` or ``SpanwiseExtractionF32``).
        _space (BsplineSpace): Underlying multi-dimensional B-spline space.
        _target (Target): Target basis tag.
        _lagrange_variant (LagrangeVariant): Point distribution used when
            ``target == "lagrange"``; ignored otherwise.
        _dtype (np.dtype): ``float32`` or ``float64``.
        _compact_ops_1d (tuple[npt.NDArray[np.float32 | np.float64], ...]):
            Per-direction compact 3D operator arrays of shape
            ``(n_compact_k, n_out_k, n_in_k)``; only non-identity rows are
            stored. Always has at least one row (a zero operator when every
            element is identity).
        _idx_maps_1d (tuple[npt.NDArray[np.intp], ...]): Per-direction compact
            index maps of shape ``(n_elements_k,)``; ``_idx_maps_1d[k][e]`` is
            the row index into ``_compact_ops_1d[k]`` for element ``e``
            (undefined for identity elements, stored as 0).
        _is_identity_mask_1d (tuple[npt.NDArray[bool], ...]): Per-direction
            identity masks of shape ``(n_elements_k,)``.
        _input_shape_per_dir (tuple[int, ...]): The columns of each direction's operator.
        _output_shape_per_dir (tuple[int, ...]): The rows of each direction's operator.
    """

    _h: Any
    _space: BsplineSpace
    _target: Target
    _lagrange_variant: LagrangeVariant
    _dtype: np.dtype[np.float32] | np.dtype[np.float64]
    _compact_ops_1d: tuple[npt.NDArray[np.float32 | np.float64], ...]
    _idx_maps_1d: tuple[npt.NDArray[np.intp], ...]
    _is_identity_mask_1d: tuple[npt.NDArray[np.bool_], ...]
    _input_shape_per_dir: tuple[int, ...]
    _output_shape_per_dir: tuple[int, ...]

    def __init__(
        self,
        space: BsplineSpace,
        target: Target,
        *,
        lagrange_variant: LagrangeVariant = LagrangeVariant.EQUISPACES,
    ) -> None:
        """Build the per-direction operators and identity masks.

        Args:
            space (BsplineSpace): Multi-dimensional B-spline space.
            target (Target): One of ``"bezier"``, ``"lagrange"``, ``"cardinal"``.
            lagrange_variant (LagrangeVariant): Point distribution used when
                ``target == "lagrange"``. Defaults to
                :attr:`pantr.basis.LagrangeVariant.EQUISPACES`.

        Raises:
            ValueError: If ``target`` is not a recognized tag, or if ``target`` is
                ``"lagrange"`` and ``lagrange_variant`` is not a
                :class:`~pantr.basis.LagrangeVariant` (or its string value).
            NotImplementedError: If any direction of ``space`` is periodic;
                periodic support is deferred to a later version.
        """
        if target not in get_args(Target):
            valid = ", ".join(repr(v) for v in get_args(Target))
            raise ValueError(f"Unknown target {target!r}; expected one of {valid}")

        dtype = np.dtype(space.dtype)
        cls = _cpp.SpanwiseExtractionF32 if dtype == _F32 else _cpp.SpanwiseExtractionF64
        variant_code = _variant_code(lagrange_variant) if target == "lagrange" else 0
        handle = cls(space._h, _TARGET_CODES[target], variant_code)
        self._space = space
        self._target = target
        self._lagrange_variant = lagrange_variant
        self._set_handle(handle)

    def _set_handle(
        self,
        h: Any,  # noqa: ANN401
    ) -> None:
        """Hold a compiled value and cache what the properties return.

        Args:
            h (Any): The compiled value.
        """
        d = self.__dict__
        d["_h"] = h
        d["_dtype"] = np.dtype(self._space.dtype)
        dim = int(h.dim)
        d["_compact_ops_1d"] = tuple(h.compact_ops(k) for k in range(dim))
        d["_idx_maps_1d"] = tuple(h.idx_map(k) for k in range(dim))
        d["_is_identity_mask_1d"] = tuple(h.identity_mask(k) for k in range(dim))
        d["_input_shape_per_dir"] = tuple(int(n) for n in h.input_shape_per_dir)
        d["_output_shape_per_dir"] = tuple(int(n) for n in h.output_shape_per_dir)
        # Drop what a previous value cached.
        d.pop("ops_1d", None)
        d.pop("num_identity_elements", None)

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle (and copy) through the constructor, with a subclass's attributes as state.

        Returns:
            tuple[Any, ...]: The rebuilding function, the class and the constructor arguments, and
            the other attributes.
        """
        return (
            _rebuild_extraction,
            (type(self), self._space, self._target, self._lagrange_variant),
            {k: v for k, v in self.__dict__.items() if k not in _VALUE_ATTRS},
        )

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore a subclass's attributes, or rebuild from a previous release's pickle.

        A previous release pickled the attributes ``_space``, ``_target``, ``_lagrange_variant``
        and its compact arrays (with whatever cached properties it held); such a state rebuilds
        the compiled value from the first three and drops the rest.

        Args:
            state (dict[str, Any]): The pickled state.
        """
        state = dict(state)
        if "_h" not in self.__dict__ and "_space" in state:
            SpanwiseElementExtraction.__init__(
                self,
                state["_space"],
                state["_target"],
                lagrange_variant=state["_lagrange_variant"],
            )
        for key in _VALUE_ATTRS:
            state.pop(key, None)
        self.__dict__.update(state)

    def __copy__(self) -> SpanwiseElementExtraction:
        """Copy: the new object shares the immutable compiled value and the space.

        Returns:
            SpanwiseElementExtraction: The copy, of the same class, with the same attributes.
        """
        new = object.__new__(type(self))
        new.__dict__.update(self.__dict__)
        return new

    def __deepcopy__(self, memo: dict[int, Any]) -> SpanwiseElementExtraction:
        """Deep copy: shares the immutable compiled value and deep-copies the rest.

        Args:
            memo (dict[int, Any]): The deep-copy memo.

        Returns:
            SpanwiseElementExtraction: The copy.
        """
        new = object.__new__(type(self))
        memo[id(self)] = new
        for key, value in self.__dict__.items():
            if key in _VALUE_ATTRS and key != "_space":
                new.__dict__[key] = value
            else:
                new.__dict__[key] = _copy.deepcopy(value, memo)
        return new

    # ---------------------------------------------------------------- properties

    @property
    def space(self) -> BsplineSpace:
        """Get the underlying B-spline space.

        Returns:
            BsplineSpace: The space supplied at construction time.
        """
        return self._space

    @property
    def target(self) -> Target:
        """Get the target basis tag.

        Returns:
            Target: One of ``"bezier"``, ``"lagrange"``, ``"cardinal"``.
        """
        return self._target

    @property
    def lagrange_variant(self) -> LagrangeVariant:
        """Get the Lagrange point distribution used for ``"lagrange"`` target.

        Returns:
            LagrangeVariant: The point distribution. Meaningless for other targets.
        """
        return self._lagrange_variant

    @property
    def dim(self) -> int:
        """Get the number of tensor-product directions.

        Returns:
            int: The dimension ``d`` of the space.
        """
        return self._space.dim

    @property
    def dtype(self) -> npt.DTypeLike:
        """Get the floating-point dtype shared by all operators.

        Returns:
            npt.DTypeLike: The dtype inherited from the space (``float32`` or ``float64``).
        """
        return self._space.dtype

    @property
    def num_intervals(self) -> tuple[int, ...]:
        """Get the per-direction number of elements (intervals).

        Returns:
            tuple[int, ...]: Length-``d`` tuple ``(n_elements_0, …, n_elements_{d-1})``.
        """
        return self._space.num_intervals

    @property
    def num_total_intervals(self) -> int:
        """Get the total number of elements across the tensor-product grid.

        Returns:
            int: ``prod(num_intervals)``.
        """
        return self._space.num_total_intervals

    @functools.cached_property
    def ops_1d(self) -> tuple[npt.NDArray[np.float32 | np.float64], ...]:
        """Get the per-direction 1D operator arrays (dense, reconstructed lazily).

        Reconstructs the full ``(n_elements_k, n_out_k, n_in_k)`` array from
        compact storage on first access and caches the result. Identity elements
        are filled with ``numpy.eye(n_out, n_in)``; non-identity elements are read
        from :attr:`compact_ops_1d`. Every extraction target builds square
        per-direction operators, and :meth:`apply` and :meth:`apply_many` refuse an
        identity-flagged operator that is not square.

        Returns:
            tuple[npt.NDArray[np.float32 | np.float64], ...]: Length-``d`` tuple
            of read-only 3D arrays; ``ops_1d[k]`` has shape
            ``(n_elements_k, n_out_k, n_in_k)``. For code that can use the
            compact layout, prefer :attr:`compact_ops_1d` and :attr:`idx_maps_1d`.
        """
        dense: list[npt.NDArray[np.float32 | np.float64]] = []
        for compact_ops, idx_map, mask in zip(
            self._compact_ops_1d, self._idx_maps_1d, self._is_identity_mask_1d, strict=True
        ):
            n_el = int(mask.shape[0])
            # shape[1] and shape[2] are direction-wide constants (same for compact and full)
            n_out, n_in = int(compact_ops.shape[1]), int(compact_ops.shape[2])
            full: npt.NDArray[np.float32 | np.float64] = np.empty(
                (n_el, n_out, n_in), dtype=compact_ops.dtype
            )
            eye = np.eye(n_out, n_in, dtype=compact_ops.dtype)
            full[mask] = eye
            full[~mask] = compact_ops[idx_map[~mask]]
            full.flags.writeable = False
            dense.append(full)
        return tuple(dense)

    @property
    def compact_ops_1d(self) -> tuple[npt.NDArray[np.float32 | np.float64], ...]:
        """Get the per-direction compact operator arrays (non-identity rows only).

        Returns:
            tuple[npt.NDArray[np.float32 | np.float64], ...]: Length-``d`` tuple
            of read-only 3D arrays; ``compact_ops_1d[k]`` has shape
            ``(n_compact_k, n_out_k, n_in_k)`` where ``n_compact_k`` is the
            number of non-identity elements in direction ``k`` (at least 1: a
            direction of identity elements keeps one zero operator that nothing
            reads). Meant to be read with :attr:`idx_maps_1d` and
            :attr:`is_identity_mask_1d`.
        """
        return self._compact_ops_1d

    @property
    def idx_maps_1d(self) -> tuple[npt.NDArray[np.intp], ...]:
        """Get the per-direction compact index maps.

        Returns:
            tuple[npt.NDArray[np.intp], ...]: Length-``d`` tuple of read-only
            1D integer arrays; ``idx_maps_1d[k]`` has shape ``(n_elements_k,)``
            and ``idx_maps_1d[k][e]`` is the row index into
            :attr:`compact_ops_1d` ``[k]`` for element ``e``. For identity
            elements the stored value is 0 (unused; the application short-circuits on
            :attr:`is_identity_mask_1d`).
        """
        return self._idx_maps_1d

    @property
    def is_identity_mask_1d(self) -> tuple[npt.NDArray[np.bool_], ...]:
        """Get the per-direction identity masks.

        All three targets use structural (multiplicity-based) identity predicates.
        For ``"bezier"``, an element is identity iff both its boundary unique knots
        have multiplicity ``>= degree + 1``; multiplicities are computed using
        ``space.tolerance``. For ``"lagrange"``, the mask delegates to the Bézier
        mask when the Lagrange-to-Bernstein matrix equals ``I`` (e.g. ``degree == 1``
        with equispaced or GLL nodes), returns all-``True`` for ``degree == 0``, and
        all-``False`` otherwise. For ``"cardinal"``, the mask is the structural output
        of :meth:`BsplineSpace1D.get_cardinal_intervals`.

        Returns:
            tuple[npt.NDArray[bool], ...]: Length-``d`` tuple of read-only
            1D boolean arrays; ``is_identity_mask_1d[k][i]`` is ``True`` iff
            the ``i``-th element in direction ``k`` has an identity operator.
        """
        return self._is_identity_mask_1d

    @property
    def input_shape_per_dir(self) -> tuple[int, ...]:
        """Get the per-direction input sizes of each element's operator.

        Returns:
            tuple[int, ...]: ``(n_in_0, …, n_in_{d-1})``.
        """
        return self._input_shape_per_dir

    @property
    def output_shape_per_dir(self) -> tuple[int, ...]:
        """Get the per-direction output sizes of each element's operator.

        Returns:
            tuple[int, ...]: ``(n_out_0, …, n_out_{d-1})``.
        """
        return self._output_shape_per_dir

    # ---------------------------------------------------------------- identity queries

    def is_identity_at(self, cell_idx: CellIndex) -> bool:
        """Check whether the per-element operator is identity along every direction.

        Args:
            cell_idx (CellIndex): Element index (flat or per-direction).

        Returns:
            bool: ``True`` iff the ``d``-dimensional operator at ``cell_idx``
            is the identity (all per-direction operators are identity).
        """
        multi = self._normalize_cell_idx(cell_idx)
        return all(bool(mask[i]) for mask, i in zip(self._is_identity_mask_1d, multi, strict=True))

    @functools.cached_property
    def num_identity_elements(self) -> int:
        """Count elements whose per-direction operators are all identity.

        Returns:
            int: The number of fully-identity elements on the tensor-product grid.
        """
        return int(self._h.num_identity_elements)

    @property
    def is_identity(self) -> bool:
        """Check whether every element on the grid has an identity operator.

        Returns:
            bool: ``True`` iff all per-direction identity masks are all-``True``,
            meaning every element's operator is the identity.
        """
        return all(bool(mask.all()) for mask in self._is_identity_mask_1d)

    def per_direction_identity_flags(self, cell_idx: CellIndex) -> tuple[bool, ...]:
        """Return the per-direction identity flags for a single element.

        Args:
            cell_idx (CellIndex): Element index (flat or per-direction).

        Returns:
            tuple[bool, ...]: Length-``d`` tuple of identity flags for the element.
        """
        multi = self._normalize_cell_idx(cell_idx)
        return tuple(
            bool(mask[i]) for mask, i in zip(self._is_identity_mask_1d, multi, strict=True)
        )

    def factors(
        self, cell_idx: CellIndex
    ) -> tuple[tuple[bool, npt.NDArray[np.float32 | np.float64] | None], ...]:
        """Return the per-direction 1D factors of one element's operator.

        The element operator is ``kron(F_0, …, F_{d-1})`` where ``F_k`` is the
        returned direction-``k`` operator, or the identity where the flag is set.
        This is the same factorization :meth:`operator` and ``__getitem__``
        use, exposed without materializing anything: identity directions yield
        ``None`` rather than an identity matrix, and non-identity directions yield
        a read-only view into :attr:`compact_ops_1d` rather than a copy. Elements
        sharing a compact row therefore share its memory -- the view object itself
        is new on each call, as numpy basic indexing always builds one, but no
        operator data is copied.

        Args:
            cell_idx (CellIndex): Element index (flat or per-direction).

        Returns:
            tuple[tuple[bool, npt.NDArray[np.float32 | np.float64] | None], ...]:
            Length-``d`` tuple whose entry ``k`` is ``(is_identity_k, op_k)``.
            ``op_k`` is ``None`` exactly when ``is_identity_k`` is ``True``;
            otherwise it is a read-only ``(n_out_k, n_in_k)`` view.

        Raises:
            IndexError: If a flat index is out of range, or a per-direction entry
                is out of range for its direction.
            ValueError: If a per-direction index has the wrong length.
            TypeError: If ``cell_idx`` is not an ``int`` or sequence of ``int``.

        Example:
            >>> import numpy as np
            >>> from pantr.bspline import BsplineSpace, BsplineSpace1D
            >>> space = BsplineSpace([BsplineSpace1D([0, 0, 0, 1, 2, 3, 4, 4, 4], 2)])
            >>> ext = SpanwiseElementExtraction(space, "cardinal")
            >>> ext.factors(1)
            ((True, None),)
            >>> flag, op = ext.factors(0)[0]
            >>> flag, op.shape, op.flags.writeable
            (False, (3, 3), False)
        """
        multi = self._normalize_cell_idx(cell_idx)
        factors: list[tuple[bool, npt.NDArray[np.float32 | np.float64] | None]] = []
        for compact, idx_map, mask, i in zip(
            self._compact_ops_1d, self._idx_maps_1d, self._is_identity_mask_1d, multi, strict=True
        ):
            if bool(mask[i]):
                factors.append((True, None))
            else:
                factors.append((False, compact[int(idx_map[i])]))
        return tuple(factors)

    # ---------------------------------------------------------------- per-cell applies

    def apply(
        self,
        v: npt.NDArray[np.float32 | np.float64],
        cell_idx: CellIndex,
        *,
        out: npt.NDArray[np.float32 | np.float64] | None = None,
        scratch: npt.NDArray[np.float32 | np.float64] | None = None,
    ) -> npt.NDArray[np.float32 | np.float64]:
        """Compute ``out = M @ v`` for the element at ``cell_idx``.

        Here ``M = kron(M_0, …, M_{d-1})`` with ``M_k`` the 1D operator at
        ``cell_idx`` in direction ``k``; identity directions short-circuit.

        Args:
            v (npt.NDArray[np.float32 | np.float64]): Input vector of shape
                ``(prod(input_shape_per_dir),)``.
            cell_idx (CellIndex): Element index (flat or per-direction).
            out (npt.NDArray[np.float32 | np.float64] | None): Optional output
                array of shape ``(prod(output_shape_per_dir),)``. Must not
                alias ``v``. Allocated
                if ``None``.
            scratch (npt.NDArray[np.float32 | np.float64] | None): Optional
                scratch buffer. Allocated if ``None``.

        Returns:
            npt.NDArray[np.float32 | np.float64]: The result array (the same
            array as ``out`` when ``out`` was provided).
        """
        return self._apply(v, cell_idx, "apply", out, scratch)

    def apply_transpose(
        self,
        v: npt.NDArray[np.float32 | np.float64],
        cell_idx: CellIndex,
        *,
        out: npt.NDArray[np.float32 | np.float64] | None = None,
        scratch: npt.NDArray[np.float32 | np.float64] | None = None,
    ) -> npt.NDArray[np.float32 | np.float64]:
        """Compute ``out = M^T @ v`` for the element at ``cell_idx``.

        Args:
            v (npt.NDArray[np.float32 | np.float64]): Input vector of shape
                ``(prod(output_shape_per_dir),)``.
            cell_idx (CellIndex): Element index.
            out (npt.NDArray[np.float32 | np.float64] | None): Optional output
                array of shape ``(prod(input_shape_per_dir),)``. Must not
                alias ``w``.
            scratch (npt.NDArray[np.float32 | np.float64] | None): Optional
                scratch buffer.

        Returns:
            npt.NDArray[np.float32 | np.float64]: The result array.
        """
        return self._apply(v, cell_idx, "apply_T", out, scratch)

    def apply_MT_K_M(
        self,
        K: npt.NDArray[np.float32 | np.float64],
        cell_idx: CellIndex,
        *,
        out: npt.NDArray[np.float32 | np.float64] | None = None,
        scratch: npt.NDArray[np.float32 | np.float64] | None = None,
    ) -> npt.NDArray[np.float32 | np.float64]:
        """Compute ``out = M^T @ K @ M`` for the element at ``cell_idx``.

        Args:
            K (npt.NDArray[np.float32 | np.float64]): Input matrix of shape
                ``(N_out, N_out)`` with ``N_out = prod(output_shape_per_dir)``.
            cell_idx (CellIndex): Element index.
            out (npt.NDArray[np.float32 | np.float64] | None): Optional output
                matrix of shape ``(N_in, N_in)`` with
                ``N_in = prod(input_shape_per_dir)``. Must not alias ``K``.
            scratch (npt.NDArray[np.float32 | np.float64] | None): Optional
                scratch buffer.

        Returns:
            npt.NDArray[np.float32 | np.float64]: The result matrix.
        """
        return self._apply(K, cell_idx, "MT_K_M", out, scratch)

    def apply_M_K_MT(
        self,
        K: npt.NDArray[np.float32 | np.float64],
        cell_idx: CellIndex,
        *,
        out: npt.NDArray[np.float32 | np.float64] | None = None,
        scratch: npt.NDArray[np.float32 | np.float64] | None = None,
    ) -> npt.NDArray[np.float32 | np.float64]:
        """Compute ``out = M @ K @ M^T`` for the element at ``cell_idx``.

        Args:
            K (npt.NDArray[np.float32 | np.float64]): Input matrix of shape
                ``(N_in, N_in)`` with ``N_in = prod(input_shape_per_dir)``.
            cell_idx (CellIndex): Element index.
            out (npt.NDArray[np.float32 | np.float64] | None): Optional output
                matrix of shape ``(N_out, N_out)`` with
                ``N_out = prod(output_shape_per_dir)``. Must not alias ``K``.
            scratch (npt.NDArray[np.float32 | np.float64] | None): Optional
                scratch buffer.

        Returns:
            npt.NDArray[np.float32 | np.float64]: The result matrix.
        """
        return self._apply(K, cell_idx, "M_K_MT", out, scratch)

    def operator(
        self,
        cell_idx: CellIndex,
        *,
        out: npt.NDArray[np.float32 | np.float64] | None = None,
    ) -> npt.NDArray[np.float32 | np.float64]:
        """Materialize the full ``(N_out, N_in)`` operator for one element.

        Assembles the full Kronecker product in memory (entries as :func:`numpy.kron`
        forms them). Prefer the matrix-free apply methods in production code when
        the full matrix is not needed explicitly.

        Args:
            cell_idx (CellIndex): Element index.
            out (npt.NDArray[np.float32 | np.float64] | None): Optional output
                matrix of shape ``(N_out, N_in)``.

        Returns:
            npt.NDArray[np.float32 | np.float64]: The full Kronecker operator.
        """
        multi = self._normalize_cell_idx(cell_idx)
        n_out = math.prod(self._output_shape_per_dir)
        n_in = math.prod(self._input_shape_per_dir)
        result = _allocate_or_validate_out(out, (n_out, n_in), self.dtype)
        # The compiled library writes a C-contiguous array; any other `out` is filled from one.
        target = result if result.flags.c_contiguous else np.empty((n_out, n_in), self._dtype)
        self._h.kron_operator(multi, target)
        if target is not result:
            result[...] = target
        return result

    def tabulate(
        self,
        *,
        out: npt.NDArray[np.float32 | np.float64] | None = None,
    ) -> npt.NDArray[np.float32 | np.float64]:
        """Materialize per-element operators for every element on the grid.

        Args:
            out (npt.NDArray[np.float32 | np.float64] | None): Optional output
                array of shape ``(num_total_intervals, N_out, N_in)``. Cells
                are ordered row-major over :attr:`num_intervals` (so flat
                index ``f`` corresponds to multi-index
                ``np.unravel_index(f, num_intervals)``).

        Returns:
            npt.NDArray[np.float32 | np.float64]: Stacked per-element operators.
        """
        n_out = math.prod(self.output_shape_per_dir)
        n_in = math.prod(self.input_shape_per_dir)
        expected = (self.num_total_intervals, n_out, n_in)
        out = _allocate_or_validate_out(out, expected, self.dtype)
        for flat in range(self.num_total_intervals):
            self.operator(flat, out=out[flat])
        return out

    # ---------------------------------------------------------------- indexing / iteration

    def __len__(self) -> int:
        """Return the total number of elements on the tensor-product grid.

        Returns:
            int: Equal to :attr:`num_total_intervals`.
        """
        return self.num_total_intervals

    def __getitem__(
        self, cell_idx: CellIndex
    ) -> tuple[tuple[npt.NDArray[np.float32 | np.float64], ...], tuple[bool, ...]]:
        """Return the per-direction operators and identity flags for one element.

        Args:
            cell_idx (CellIndex): Element index (flat or per-direction).

        Returns:
            tuple[tuple[npt.NDArray[np.float32 | np.float64], ...], tuple[bool, ...]]:
            ``(ops_for_cell, identity_flags)`` where ``ops_for_cell[k]`` is a
            2D array of shape ``(n_out_k, n_in_k)`` (identity elements return a
            fresh ``numpy.eye``; non-identity elements return a row from
            :attr:`compact_ops_1d`) and ``identity_flags`` is the per-direction
            identity mask at this element.
        """
        ops, flags = self._ops_and_flags_for_cell(self._normalize_cell_idx(cell_idx))
        return ops, flags

    def __iter__(
        self,
    ) -> Iterator[tuple[tuple[npt.NDArray[np.float32 | np.float64], ...], tuple[bool, ...]]]:
        """Iterate over all elements in row-major order over :attr:`num_intervals`.

        Yields:
            tuple[tuple[npt.NDArray[np.float32 | np.float64], ...], tuple[bool, ...]]:
            Same shape as :meth:`__getitem__`'s return value.
        """
        for flat in range(self.num_total_intervals):
            yield self[flat]

    # ---------------------------------------------------------------- internals

    def _normalize_cell_idx(self, cell_idx: CellIndex) -> tuple[int, ...]:
        """Convert a flat or per-direction index into a validated per-direction tuple.

        Args:
            cell_idx (CellIndex): Flat ``int`` or per-direction sequence.

        Returns:
            tuple[int, ...]: Length-``d`` tuple of non-negative element indices.

        Raises:
            IndexError: If a flat index is out of range (negative indices are
                not supported and are also rejected), or a per-direction entry
                is out of range for its direction.
            ValueError: If a per-direction index has the wrong length.
            TypeError: If ``cell_idx`` is not an ``int`` or sequence of ``int``.
        """
        num_intervals = self.num_intervals
        d = len(num_intervals)
        if isinstance(cell_idx, int | np.integer):
            flat = int(cell_idx)
            total = self.num_total_intervals
            if flat < 0 or flat >= total:
                raise IndexError(f"Flat cell index {flat} out of range for {total} elements")
            multi = np.unravel_index(flat, num_intervals)
            return tuple(int(i) for i in multi)
        if isinstance(cell_idx, tuple | list | np.ndarray):
            seq = tuple(int(x) for x in cell_idx)
            if len(seq) != d:
                raise ValueError(f"Per-direction cell index has length {len(seq)}, expected {d}")
            for k, (i, n) in enumerate(zip(seq, num_intervals, strict=True)):
                if i < 0 or i >= n:
                    raise IndexError(
                        f"Cell index {i} out of range for direction {k} with {n} elements"
                    )
            return seq
        raise TypeError(f"cell_idx must be int or sequence of int; got {type(cell_idx).__name__}")

    def _ops_and_flags_for_cell(
        self, multi: tuple[int, ...]
    ) -> tuple[
        tuple[npt.NDArray[np.float32 | np.float64], ...],
        tuple[bool, ...],
    ]:
        """Extract per-direction operators and identity flags from compact storage.

        Args:
            multi (tuple[int, ...]): Already-validated per-direction element indices.

        Returns:
            tuple[tuple[npt.NDArray, ...], tuple[bool, ...]]: ``(ops, flags)``
            where each ``ops[k]`` is a 2D ``(n_out_k, n_in_k)`` array from
            compact storage (or a fresh ``numpy.eye`` for identity elements).
        """
        ops: list[npt.NDArray[np.float32 | np.float64]] = []
        flags: list[bool] = []
        for compact, idx_map, mask, i in zip(
            self._compact_ops_1d, self._idx_maps_1d, self._is_identity_mask_1d, multi, strict=True
        ):
            n_out, n_in = int(compact.shape[1]), int(compact.shape[2])
            is_id = bool(mask[i])
            flags.append(is_id)
            if is_id:
                ops.append(np.eye(n_out, n_in, dtype=compact.dtype))
            else:
                ops.append(compact[int(idx_map[i])])
        return tuple(ops), tuple(flags)

    def _apply(
        self,
        operand: npt.NDArray[np.float32 | np.float64],
        cell_idx: CellIndex,
        op_kind: OpKind,
        out: npt.NDArray[np.float32 | np.float64] | None,
        scratch: npt.NDArray[np.float32 | np.float64] | None,
    ) -> npt.NDArray[np.float32 | np.float64]:
        """Dispatch a single-element apply variant through the Layer-2 helper.

        Args:
            operand (npt.NDArray[np.float32 | np.float64]): Input vector or
                matrix (shape depends on ``op_kind``).
            cell_idx (CellIndex): Element index.
            op_kind (OpKind): Which apply variant to dispatch.
            out (npt.NDArray[np.float32 | np.float64] | None): Optional output.
            scratch (npt.NDArray[np.float32 | np.float64] | None): Optional scratch.

        Returns:
            npt.NDArray[np.float32 | np.float64]: The result array.
        """
        multi = self._normalize_cell_idx(cell_idx)
        in_shape, out_shape = _operation_shapes(
            self._input_shape_per_dir, self._output_shape_per_dir, op_kind
        )
        _validate_operand(operand, in_shape, self._dtype)
        result = _allocate_or_validate_out(out, out_shape, self._dtype)
        # The compiled library reads and writes C-contiguous arrays; any other operand is
        # copied, and any other `out` is filled from a temporary.
        target = result if result.flags.c_contiguous else np.empty(out_shape, self._dtype)
        operand_c = operand if operand.flags.c_contiguous else np.ascontiguousarray(operand)
        scratch_c = _scratch_argument(scratch, 1, self._dtype)
        getattr(self._h, _ONE_CELL_METHODS[op_kind])(operand_c, multi, target, scratch_c)
        if target is not result:
            result[...] = target
        return result

    def _apply_many(
        self,
        operand: npt.NDArray[np.float32 | np.float64],
        cell_indices: CellIndicesBatch,
        op_kind: OpKind,
        out: npt.NDArray[np.float32 | np.float64] | None,
        scratch: npt.NDArray[np.float32 | np.float64] | None,
    ) -> npt.NDArray[np.float32 | np.float64]:
        """Run a batch apply variant in the compiled library.

        Args:
            operand (npt.NDArray[np.float32 | np.float64]): Batch input array.
            cell_indices (CellIndicesBatch): Flat or per-direction cell indices.
            op_kind (OpKind): Which apply variant to run.
            out (npt.NDArray[np.float32 | np.float64] | None): Optional output.
            scratch (npt.NDArray[np.float32 | np.float64] | None): Optional scratch.

        Returns:
            npt.NDArray[np.float32 | np.float64]: Result array; shape is
            ``(n_cells, N_out)`` for ``"apply"``, ``(n_cells, N_in)`` for
            ``"apply_T"``, ``(n_cells, N_in, N_in)`` for ``"MT_K_M"``, or
            ``(n_cells, N_out, N_out)`` for ``"M_K_MT"``.

        Raises:
            IndexError: If any cell index is out of range.
            ValueError: If operand shape/dtype or ``out``/``scratch`` are invalid.
            TypeError: If ``cell_indices`` contains non-integer values.
        """
        idx2d = normalize_cell_indices(cell_indices, self.num_intervals)
        n_cells = int(idx2d.shape[0])
        per_cell_in, per_cell_out = _operation_shapes(
            self._input_shape_per_dir, self._output_shape_per_dir, op_kind
        )
        expected_operand = (n_cells, *per_cell_in)
        expected_out = (n_cells, *per_cell_out)
        _validate_operand(operand, expected_operand, self._dtype)
        result = _allocate_or_validate_out(out, expected_out, self._dtype)
        target = result if result.flags.c_contiguous else np.empty(expected_out, self._dtype)
        operand_c = operand if operand.flags.c_contiguous else np.ascontiguousarray(operand)
        cells = np.ascontiguousarray(idx2d, dtype=np.int64)
        scratch_c = _scratch_argument(scratch, 2, self._dtype)
        getattr(self._h, _BATCH_METHODS[op_kind])(operand_c, cells, target, scratch_c)
        if target is not result:
            result[...] = target
        return result

    # ---------------------------------------------------------------- batch applies

    def apply_many(
        self,
        v: npt.NDArray[np.float32 | np.float64],
        cell_indices: CellIndicesBatch,
        *,
        out: npt.NDArray[np.float32 | np.float64] | None = None,
        scratch: npt.NDArray[np.float32 | np.float64] | None = None,
    ) -> npt.NDArray[np.float32 | np.float64]:
        """Compute ``out[c] = M_c @ v[c]`` for all cells in the batch.

        ``M_c = kron(M_0[c_0], …, M_{d-1}[c_{d-1}])`` with ``M_k[c_k]`` the
        1D operator at element ``c_k`` in direction ``k``; identity directions
        short-circuit per cell.

        Args:
            v (npt.NDArray[np.float32 | np.float64]): Batch input vectors,
                shape ``(n_cells, N_in)`` with
                ``N_in = prod(input_shape_per_dir)``.
            cell_indices (CellIndicesBatch): Cell indices — flat 1-D array of
                shape ``(n_cells,)`` or per-direction 2-D array of shape
                ``(n_cells, d)``.
            out (npt.NDArray[np.float32 | np.float64] | None): Optional output
                array of shape ``(n_cells, N_out)``. Must not alias ``v``.
                Allocated if ``None``.
            scratch (npt.NDArray[np.float32 | np.float64] | None): Optional
                per-cell scratch array of shape ``(n_cells, s)`` with
                ``s >= scratch_size_per_cell``. Allocated if ``None``.

        Returns:
            npt.NDArray[np.float32 | np.float64]: Result array of shape
            ``(n_cells, N_out)``.
        """
        return self._apply_many(v, cell_indices, "apply", out, scratch)

    def apply_transpose_many(
        self,
        v: npt.NDArray[np.float32 | np.float64],
        cell_indices: CellIndicesBatch,
        *,
        out: npt.NDArray[np.float32 | np.float64] | None = None,
        scratch: npt.NDArray[np.float32 | np.float64] | None = None,
    ) -> npt.NDArray[np.float32 | np.float64]:
        """Compute ``out[c] = M_c^T @ v[c]`` for all cells in the batch.

        Args:
            v (npt.NDArray[np.float32 | np.float64]): Batch input vectors,
                shape ``(n_cells, N_out)`` with
                ``N_out = prod(output_shape_per_dir)``.
            cell_indices (CellIndicesBatch): Cell indices — flat or per-direction.
            out (npt.NDArray[np.float32 | np.float64] | None): Optional output
                array of shape ``(n_cells, N_in)``. Must not alias ``w``.
                Allocated if ``None``.
            scratch (npt.NDArray[np.float32 | np.float64] | None): Optional
                per-cell scratch array. Allocated if ``None``.

        Returns:
            npt.NDArray[np.float32 | np.float64]: Result array of shape
            ``(n_cells, N_in)``.
        """
        return self._apply_many(v, cell_indices, "apply_T", out, scratch)

    def apply_MT_K_M_many(
        self,
        K: npt.NDArray[np.float32 | np.float64],
        cell_indices: CellIndicesBatch,
        *,
        out: npt.NDArray[np.float32 | np.float64] | None = None,
        scratch: npt.NDArray[np.float32 | np.float64] | None = None,
    ) -> npt.NDArray[np.float32 | np.float64]:
        """Compute ``out[c] = M_c^T @ K[c] @ M_c`` for all cells in the batch.

        Args:
            K (npt.NDArray[np.float32 | np.float64]): Batch input matrices,
                shape ``(n_cells, N_out, N_out)`` with
                ``N_out = prod(output_shape_per_dir)``.
            cell_indices (CellIndicesBatch): Cell indices — flat or per-direction.
            out (npt.NDArray[np.float32 | np.float64] | None): Optional output
                array of shape ``(n_cells, N_in, N_in)``. Must not alias ``K``.
                Allocated if ``None``.
            scratch (npt.NDArray[np.float32 | np.float64] | None): Optional
                per-cell scratch array. Allocated if ``None``.

        Returns:
            npt.NDArray[np.float32 | np.float64]: Result array of shape
            ``(n_cells, N_in, N_in)``.
        """
        return self._apply_many(K, cell_indices, "MT_K_M", out, scratch)

    def apply_M_K_MT_many(
        self,
        K: npt.NDArray[np.float32 | np.float64],
        cell_indices: CellIndicesBatch,
        *,
        out: npt.NDArray[np.float32 | np.float64] | None = None,
        scratch: npt.NDArray[np.float32 | np.float64] | None = None,
    ) -> npt.NDArray[np.float32 | np.float64]:
        """Compute ``out[c] = M_c @ K[c] @ M_c^T`` for all cells in the batch.

        Args:
            K (npt.NDArray[np.float32 | np.float64]): Batch input matrices,
                shape ``(n_cells, N_in, N_in)`` with
                ``N_in = prod(input_shape_per_dir)``.
            cell_indices (CellIndicesBatch): Cell indices — flat or per-direction.
            out (npt.NDArray[np.float32 | np.float64] | None): Optional output
                array of shape ``(n_cells, N_out, N_out)``. Must not alias ``K``.
                Allocated if ``None``.
            scratch (npt.NDArray[np.float32 | np.float64] | None): Optional
                per-cell scratch array. Allocated if ``None``.

        Returns:
            npt.NDArray[np.float32 | np.float64]: Result array of shape
            ``(n_cells, N_out, N_out)``.
        """
        return self._apply_many(K, cell_indices, "M_K_MT", out, scratch)


def normalize_cell_indices(
    cell_indices: CellIndicesBatch,
    num_intervals: tuple[int, ...],
) -> npt.NDArray[np.intp]:
    """Convert batch cell indices to a validated ``(n_cells, d)`` integer array.

    Accepts flat indices (row-major over ``num_intervals``) or per-direction
    indices, in array or list form. Validates that all values are within their
    respective per-direction bounds and that the shape is consistent.

    Args:
        cell_indices (CellIndicesBatch): Flat 1-D array or list of ``n_cells``
            flat indices, or 2-D array/list of shape ``(n_cells, d)`` with
            per-direction indices.
        num_intervals (tuple[int, ...]): Per-direction element counts
            ``(n_el_0, …, n_el_{d-1})``.

    Returns:
        npt.NDArray[np.intp]: 2-D integer array of shape ``(n_cells, d)``.

    Raises:
        IndexError: If any value is out of range for its per-direction bound.
        ValueError: If ``cell_indices`` has the wrong shape or ndim.
        TypeError: If ``cell_indices`` contains non-integer values.
    """
    d = len(num_intervals)
    arr_raw = np.asarray(cell_indices)
    if arr_raw.size > 0 and not np.issubdtype(arr_raw.dtype, np.integer):
        raise TypeError(f"cell_indices must contain integers; got dtype {arr_raw.dtype}")
    arr = arr_raw.astype(np.intp)
    if arr.ndim == 1:
        n_cells = arr.shape[0]
        total = 1
        for n in num_intervals:
            total *= n
        if n_cells > 0 and (int(arr.min()) < 0 or int(arr.max()) >= total):
            raise IndexError(
                f"Flat cell indices must be in [0, {total}); "
                f"got range [{int(arr.min())}, {int(arr.max())}]"
            )
        rows = np.unravel_index(arr, num_intervals)
        return np.stack(rows, axis=1)
    if arr.ndim == 2:  # noqa: PLR2004
        if arr.shape[1] != d:
            raise ValueError(
                f"Per-direction cell_indices must have shape (n_cells, {d}); got shape {arr.shape}"
            )
        n_cells = arr.shape[0]
        if n_cells > 0:
            for k, n_el in enumerate(num_intervals):
                col = arr[:, k]
                if int(col.min()) < 0 or int(col.max()) >= n_el:
                    raise IndexError(
                        f"cell_indices[:, {k}] must be in [0, {n_el}); "
                        f"got range [{int(col.min())}, {int(col.max())}]"
                    )
        return arr
    raise ValueError(f"cell_indices must be 1-D (flat) or 2-D (per-direction); got ndim={arr.ndim}")


def _rebuild_extraction(
    cls: type[SpanwiseElementExtraction],
    space: BsplineSpace,
    target: Target,
    lagrange_variant: LagrangeVariant,
) -> SpanwiseElementExtraction:
    """Rebuild a pickled extraction of class ``cls``, bypassing ``cls.__init__``.

    A subclass's constructor may take other arguments; the previous release's pickles did not
    call it either.

    Args:
        cls (type[SpanwiseElementExtraction]): The class of the pickled extraction.
        space (BsplineSpace): Its space.
        target (Target): Its target basis.
        lagrange_variant (LagrangeVariant): Its Lagrange point distribution.

    Returns:
        SpanwiseElementExtraction: The extraction (its extra attributes are restored by
        ``__setstate__``).
    """
    ext_obj = cls.__new__(cls)
    SpanwiseElementExtraction.__init__(ext_obj, space, target, lagrange_variant=lagrange_variant)
    return ext_obj


def _validate_op_kind(op_kind: str) -> OpKind:
    """Validate an operation-kind tag.

    Args:
        op_kind (str): Tag to validate.

    Returns:
        OpKind: The validated tag (narrowed to :data:`OpKind`).

    Raises:
        ValueError: If ``op_kind`` is not one of :data:`OpKind`'s literals.
    """
    if op_kind not in get_args(OpKind):
        valid = ", ".join(repr(v) for v in get_args(OpKind))
        raise ValueError(f"Unknown op_kind {op_kind!r}; expected one of {valid}")
    return cast(OpKind, op_kind)


def _operation_shapes(
    input_shape_per_dir: tuple[int, ...],
    output_shape_per_dir: tuple[int, ...],
    op_kind: OpKind,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return ``(input_shape, output_shape)`` for a given operation kind.

    For vector apply variants the shapes are 1D ``(N,)`` tuples. For bilateral
    variants they are 2D ``(N, N)`` tuples. ``N`` is the product of the
    corresponding per-direction sizes.

    Args:
        input_shape_per_dir (tuple[int, ...]): Per-direction input sizes.
        output_shape_per_dir (tuple[int, ...]): Per-direction output sizes.
        op_kind (OpKind): Operation kind.

    Returns:
        tuple[tuple[int, ...], tuple[int, ...]]: ``(input_shape, output_shape)``
        of the operand and result arrays.
    """
    _validate_op_kind(op_kind)
    n_in = math.prod(input_shape_per_dir)
    n_out = math.prod(output_shape_per_dir)
    if op_kind == "apply":
        return (n_in,), (n_out,)
    if op_kind == "apply_T":
        return (n_out,), (n_in,)
    if op_kind == "MT_K_M":
        return (n_out, n_out), (n_in, n_in)
    # op_kind == "M_K_MT"
    return (n_in, n_in), (n_out, n_out)


def _validate_operand(
    operand: npt.NDArray[Any],
    expected_shape: tuple[int, ...],
    dtype: npt.DTypeLike,
) -> None:
    """Validate shape and dtype of an input operand.

    Args:
        operand (np.ndarray): Input array to check.
        expected_shape (tuple[int, ...]): Required shape.
        dtype (npt.DTypeLike): Required dtype.

    Raises:
        ValueError: If the operand shape or dtype does not match.
    """
    if operand.shape != expected_shape:
        raise ValueError(f"Operand has shape {operand.shape}, but expected shape {expected_shape}")
    if operand.dtype != np.dtype(dtype):
        raise ValueError(f"Operand has dtype {operand.dtype}, but expected dtype {np.dtype(dtype)}")


def _scratch_argument(
    scratch: npt.NDArray[np.float32 | np.float64] | None,
    ndim: int,
    dtype: np.dtype[Any],
) -> npt.NDArray[np.float32 | np.float64] | None:
    """Check the Python-visible properties of a scratch array and make it fit the library.

    The scratch's size is checked by the library, with the same messages. A scratch that is not
    C-contiguous is replaced by a C-contiguous array of its shape: the contents of a scratch are
    never read.

    Args:
        scratch (npt.NDArray[np.float32 | np.float64] | None): The caller's scratch, or ``None``.
        ndim (int): ``1`` for a one-cell application, ``2`` for a batch (a row per cell).
        dtype (np.dtype): The dtype of the extraction.

    Returns:
        npt.NDArray[np.float32 | np.float64] | None: The array to hand to the library, or
        ``None`` if there is no scratch.

    Raises:
        ValueError: If ``scratch`` has another number of dimensions or another dtype, or is not
            writeable.
    """
    if scratch is None:
        return None
    batch = ndim == 2  # noqa: PLR2004
    prefix = "Batch scratch" if batch else "Scratch"
    if scratch.ndim != ndim:
        raise ValueError(f"{prefix} must be {ndim}D; got ndim={scratch.ndim}")
    if scratch.dtype != dtype:
        expected = f"{dtype}" if batch else f"dtype {dtype}"
        raise ValueError(f"{prefix} has dtype {scratch.dtype}, but expected {expected}")
    if not scratch.flags.writeable:
        raise ValueError(f"{prefix} array is not writeable")
    if scratch.flags.c_contiguous:
        return scratch
    return np.empty(scratch.shape, dtype=dtype)


# The op_kind shape helper is public so callers can build operands of the right shape.
def operand_shape(
    extraction: SpanwiseElementExtraction, op_kind: OpKind
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return the expected ``(input_shape, output_shape)`` for an apply variant.

    Args:
        extraction (SpanwiseElementExtraction): Extraction object supplying
            per-direction shapes.
        op_kind (OpKind): One of the :data:`OpKind` literals: ``"apply"``,
            ``"apply_T"``, ``"MT_K_M"``, ``"M_K_MT"``.

    Returns:
        tuple[tuple[int, ...], tuple[int, ...]]: ``(input_shape, output_shape)``.
    """
    return _operation_shapes(
        extraction.input_shape_per_dir, extraction.output_shape_per_dir, op_kind
    )


class ExtractionStructView(NamedTuple):
    """Immutable struct view of a :class:`SpanwiseElementExtraction`.

    A :class:`typing.NamedTuple` bundling the compact per-direction operator
    storage, index maps, identity masks, and shape metadata into a single
    object: the same bundle the extraction exposes member by member, for a
    consumer that wants to carry it around as one value. The array fields are
    read-only views of the extraction's compiled value, and the tuple fields
    hold one entry per direction (the per-direction shapes may differ).

    Construct via :func:`make_struct_view`. Field semantics mirror the
    same-named members of :class:`SpanwiseElementExtraction`:

    - ``compact_ops_1d`` — per-direction compact 3D operator arrays of shape
      ``(n_compact_k, n_out_k, n_in_k)``; only non-identity rows are stored.
      Always has at least one row (a zero operator when every element is
      identity).
    - ``idx_maps_1d`` — per-direction compact index maps of shape
      ``(n_elements_k,)``.
    - ``is_identity_mask_1d`` — per-direction identity masks of shape
      ``(n_elements_k,)``.
    - ``num_intervals`` — per-direction number of elements.
    - ``input_shape_per_dir`` — per-direction input sizes
      ``(n_in_0, …, n_in_{d-1})``.
    - ``output_shape_per_dir`` — per-direction output sizes
      ``(n_out_0, …, n_out_{d-1})``.
    - ``dim`` — number of tensor-product directions ``d``.
    """

    compact_ops_1d: tuple[npt.NDArray[np.float32 | np.float64], ...]
    idx_maps_1d: tuple[npt.NDArray[np.intp], ...]
    is_identity_mask_1d: tuple[npt.NDArray[np.bool_], ...]
    num_intervals: tuple[int, ...]
    input_shape_per_dir: tuple[int, ...]
    output_shape_per_dir: tuple[int, ...]
    dim: int


def make_struct_view(extraction: SpanwiseElementExtraction) -> ExtractionStructView:
    """Bundle a :class:`SpanwiseElementExtraction` into a struct view.

    Shares the underlying per-direction arrays (no copies). The arrays are
    read-only views of the extraction's compiled value, so the returned view
    is safe to pass around without risk of accidental mutation.

    Args:
        extraction (SpanwiseElementExtraction): Source extraction object.

    Returns:
        ExtractionStructView: Named tuple wrapping the extraction's compact
        storage and shape metadata.

    Example:
        >>> from pantr.bspline import BsplineSpace1D, BsplineSpace
        >>> from pantr.bspline import SpanwiseElementExtraction, make_struct_view
        >>> sp = BsplineSpace1D([0, 0, 0, 1, 2, 2, 2], 2)
        >>> space = BsplineSpace([sp, sp])
        >>> ext = SpanwiseElementExtraction(space, "bezier")
        >>> view = make_struct_view(ext)
        >>> view.dim
        2
    """
    return ExtractionStructView(
        compact_ops_1d=extraction.compact_ops_1d,
        idx_maps_1d=extraction.idx_maps_1d,
        is_identity_mask_1d=extraction.is_identity_mask_1d,
        num_intervals=tuple(int(n) for n in extraction.num_intervals),
        input_shape_per_dir=extraction.input_shape_per_dir,
        output_shape_per_dir=extraction.output_shape_per_dir,
        dim=int(extraction.dim),
    )


__all__ = [
    "CellIndex",
    "CellIndicesBatch",
    "ExtractionStructView",
    "SpanwiseElementExtraction",
    "Target",
    "make_struct_view",
    "normalize_cell_indices",
    "operand_shape",
]
