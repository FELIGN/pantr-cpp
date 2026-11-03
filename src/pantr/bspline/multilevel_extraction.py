r"""Multi-level Bézier extraction for truncated hierarchical B-spline spaces.

This module provides :class:`MultiLevelExtraction`, the hierarchical counterpart of
:class:`~pantr.bspline.SpanwiseElementExtraction`.  It exposes, per active cell, the
*multi-level extraction operator* and the *multi-level Bézier extraction operator* of
D'Angella et al. (2017) / D'Angella (2021, ch. 4), which flatten the (truncated)
hierarchical basis on an element into a fixed single-level reference basis.

On an active cell :math:`\epsilon` of level ``L`` the active hierarchical functions
:math:`H^\epsilon` are a linear combination of the level-``L`` tensor-product B-splines
:math:`N^{\epsilon,L}` with support on :math:`\epsilon` (the *multi-level extraction
operator* :math:`M^\epsilon`), and composing with the standard per-element Bézier
extraction :math:`E^\epsilon` gives the *multi-level Bézier extraction* :math:`C^\epsilon`:

.. math::
    H^\epsilon = M^\epsilon N^{\epsilon,L} = M^\epsilon E^\epsilon B = C^\epsilon B,

mapping a fixed Bernstein reference basis ``B`` (on :math:`[0, 1]^d`) to the active
hierarchical functions on the cell.

Note:
    :math:`M^\epsilon` is built from the space's already-truncated coefficients (the
    Giannelli-Jüttler-Speleers truncation, which keeps and refines forward the passive
    functions that straddle a refinement boundary), so it is correct on narrow refinement
    bands.  It does **not** use the activeness-restricted local truncation of
    D'Angella et al. (2017, §3.6.1), which drops such functions; see Eq. 4.7 of the 2021
    thesis for the corrected predicate.

Main exports:

- :class:`MultiLevelExtraction`: per-element multi-level (Bézier) extraction operators.
"""

from __future__ import annotations

import copy
from typing import Any, cast, get_args

import numpy as np
import numpy.typing as npt

from .._ext import ext
from ..basis._basis_utils import _allocate_or_validate_out
from ._thb_spline_space import THBSplineSpace
from .spanwise_element_extraction import _TARGET_CODES, Target

_cpp: Any = ext.bspline
"""The compiled multi-level extractions (private; untyped: one class per dtype)."""

_F32 = np.dtype(np.float32)


class MultiLevelExtraction:
    r"""Per-element multi-level (Bézier) extraction for a :class:`THBSplineSpace`.

    Mirrors :class:`~pantr.bspline.SpanwiseElementExtraction`: it is constructed from a
    space and a ``target`` reference basis, builds the single-level per-level extractions,
    and exposes per-element operators via :meth:`operator`.  Because hierarchical
    refinement introduces a non-constant number of active functions per cell (and the
    hierarchical basis is not of tensor-product structure), the operators are ragged
    across cells; there is consequently no constant-shape ``tabulate`` / ``ops_1d``.

    For a cell with ``K = active_basis(cid).size`` active functions, degree ``p``, and
    dimension ``d`` (so ``n = (p + 1) ** d`` single-level functions on the cell):

    - :meth:`multilevel_operator` returns :math:`M^\epsilon` of shape ``(K, n)`` mapping
      the level-``L`` tensor-product B-splines on the cell to the active hierarchical
      functions (independent of ``target``).
    - :meth:`operator` returns :math:`C^\epsilon = M^\epsilon E^\epsilon` of shape
      ``(K, n)`` mapping the ``target`` reference basis (Bernstein on :math:`[0, 1]^d`
      for ``"bezier"``) to the active hierarchical functions.

    The operators' rows are ordered as :meth:`active_basis` (sorted global dof).

    References:
        Multi-level Bézier extraction for hierarchical local refinement
        :cite:p:`dangella2018multilevel`.

    Attributes:
        _h (Any): The compiled extraction (``MultiLevelExtractionF64`` or
            ``MultiLevelExtractionF32``); it computes each function's coefficients in the finer
            levels' bases once, for every cell.
        _space (THBSplineSpace): The hierarchical space being extracted.
        _target (Target): The single-level reference basis tag.
    """

    __slots__ = ("_h", "_space", "_target")

    def __init__(self, space: THBSplineSpace, target: Target = "bezier") -> None:
        """Create a multi-level extraction for a hierarchical space.

        Args:
            space (THBSplineSpace): The truncated (or non-truncated) hierarchical space.
            target (Target): Single-level reference basis, one of ``"bezier"``,
                ``"lagrange"``, ``"cardinal"``.  Defaults to ``"bezier"``.

        Raises:
            TypeError: If ``space`` is not a :class:`THBSplineSpace`.
            ValueError: If ``target`` is not a recognized tag.
        """
        if not isinstance(space, THBSplineSpace):
            raise TypeError(f"space must be a THBSplineSpace; got {type(space).__name__!r}.")
        if target not in get_args(Target):
            valid = ", ".join(repr(v) for v in get_args(Target))
            raise ValueError(f"Unknown target {target!r}; expected one of {valid}.")
        if space._root_space.dtype == _F32:
            self._h = _cpp.MultiLevelExtractionF32(space._h, _TARGET_CODES[target])
        else:
            self._h = _cpp.MultiLevelExtractionF64(space._h, _TARGET_CODES[target])
        self._space = space
        self._target = target

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle through the constructor arguments, with a subclass's attributes as state.

        Returns:
            tuple[Any, ...]: The rebuilding function, its arguments and the extra attributes.
        """
        return (
            _rebuild_extraction,
            (type(self), self._space, self._target),
            dict(self.__dict__) if hasattr(self, "__dict__") else None,
        )

    def __setstate__(
        self, state: dict[str, Any] | tuple[dict[str, Any] | None, dict[str, Any]]
    ) -> None:
        """Restore a subclass's attributes, or rebuild from a previous release's pickle.

        A previous release pickled the slots of a plain object, a pair ``(attributes, slots)``
        with the attributes of a subclass (or ``None``) and the slot values. Such a state rebuilds
        the compiled extraction from ``_space`` and ``_target`` and drops the other slots (the
        Oslo matrices and the caches).

        Args:
            state (dict[str, Any] | tuple[dict[str, Any] | None, dict[str, Any]]): The pickled
                state: a subclass's extra attributes, or the previous release's pair.
        """
        if isinstance(state, tuple):
            extra, slots = state
            MultiLevelExtraction.__init__(self, slots["_space"], slots["_target"])
            state = extra or {}
        if state:
            self.__dict__.update(state)

    def __copy__(self) -> MultiLevelExtraction:
        """Copy: the new object shares the compiled extraction and the space.

        Returns:
            MultiLevelExtraction: The copy, of the same class, with the same attributes.
        """
        new = object.__new__(type(self))
        new._h = self._h
        new._space = self._space
        new._target = self._target
        if hasattr(self, "__dict__"):
            new.__dict__.update(self.__dict__)
        return new

    def __deepcopy__(self, memo: dict[int, Any]) -> MultiLevelExtraction:
        """Deep copy: shares the compiled extraction and deep-copies the space.

        Args:
            memo (dict[int, Any]): The deep-copy memo.

        Returns:
            MultiLevelExtraction: The copy.
        """
        new = object.__new__(type(self))
        memo[id(self)] = new
        new._h = self._h
        new._space = copy.deepcopy(self._space, memo)
        new._target = self._target
        if hasattr(self, "__dict__"):
            for key, value in self.__dict__.items():
                new.__dict__[key] = copy.deepcopy(value, memo)
        return new

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def space(self) -> THBSplineSpace:
        """Get the underlying hierarchical space.

        Returns:
            THBSplineSpace: The space supplied at construction time.
        """
        return self._space

    @property
    def target(self) -> Target:
        """Get the single-level reference basis tag.

        Returns:
            Target: One of ``"bezier"``, ``"lagrange"``, ``"cardinal"``.
        """
        return self._target

    @property
    def dim(self) -> int:
        """Get the parametric dimension.

        Returns:
            int: Number of parametric directions.
        """
        return self._space.dim

    @property
    def dtype(self) -> type[np.float64]:
        """Get the floating-point dtype of the operators.

        Returns:
            type[np.float64]: Always ``numpy.float64``; Oslo matrices and all operators
            are computed in double precision.
        """
        return np.float64

    @property
    def num_elements(self) -> int:
        """Get the number of active cells (elements).

        Returns:
            int: ``space.grid.num_cells``.
        """
        return self._space.grid.num_cells

    def __len__(self) -> int:
        """Return the number of active cells.

        Returns:
            int: ``num_elements``.
        """
        return self.num_elements

    # ------------------------------------------------------------------
    # Per-element operators
    # ------------------------------------------------------------------

    def active_basis(self, cid: int) -> npt.NDArray[np.int64]:
        """Return the global dofs labelling the rows of the operators on cell ``cid``.

        Args:
            cid (int): Active cell flat id in ``[0, num_elements)``.

        Returns:
            npt.NDArray[np.int64]: Sorted global hierarchical-dof indices (the operator
            rows), as returned by :meth:`THBSplineSpace.active_basis`.

        Raises:
            IndexError: If ``cid`` is out of range.
            RuntimeError: If the grid has been modified since construction.
        """
        return self._space.active_basis(cid)

    def _operator(
        self,
        cid: int,
        out: npt.NDArray[np.float64] | None,
        method: str,
    ) -> npt.NDArray[np.float64]:
        """Fill the rows of one of the compiled operators on cell ``cid``.

        Args:
            cid (int): Active cell flat id in ``[0, num_elements)``.
            out (npt.NDArray[np.float64] | None): The caller's output array, or ``None``.
            method (str): The compiled method: ``"multilevel_operator"`` or
                ``"extraction_operator"``.

        Returns:
            npt.NDArray[np.float64]: The operator, in ``out`` when given.

        Raises:
            IndexError: If ``cid`` is out of range.
            ValueError: If ``out`` has the wrong shape, dtype, or is not writeable.
            RuntimeError: If the grid has been modified since construction.
        """
        space = self._space
        space._check_not_stale()
        cell = space._cell_id(cid)
        n_active = int(self._h.num_active_on(cell))
        # Every target's single-level operator is square: ``prod(degree + 1)`` columns.
        n_cols = int(np.prod([p + 1 for p in space.degrees]))
        result = cast(
            npt.NDArray[np.float64],
            _allocate_or_validate_out(out, (n_active, n_cols), np.float64),
        )
        # The compiled call writes a C-contiguous array; another layout is filled from one.
        target = result if result.flags.c_contiguous else np.empty((n_active, n_cols), np.float64)
        getattr(self._h, method)(cell, target)
        if target is not result:
            result[...] = target
        return result

    def multilevel_operator(
        self,
        cid: int,
        *,
        out: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        r"""Return the multi-level extraction operator :math:`M^\epsilon` on cell ``cid``.

        :math:`M^\epsilon` (shape ``(K, n)``) maps the level-``L`` tensor-product
        B-splines with support on the cell to the active hierarchical functions
        (``H^\epsilon = M^\epsilon N^{\epsilon,L}``).  Rows follow :meth:`active_basis`;
        columns are the ``(p + 1) ** d`` single-level functions on the cell in C-order.

        Args:
            cid (int): Active cell flat id in ``[0, num_elements)``.
            out (npt.NDArray[np.float64] | None): Optional output array of shape
                ``(K, n)`` where ``K = active_basis(cid).size`` and
                ``n = (p + 1) ** d``.  Allocated when ``None``.

        Returns:
            npt.NDArray[np.float64]: The operator :math:`M^\epsilon`.

        Raises:
            IndexError: If ``cid`` is out of range.
            ValueError: If ``out`` has the wrong shape, dtype, or is not writeable.
            RuntimeError: If the grid has been modified since construction.
        """
        return self._operator(cid, out, "multilevel_operator")

    def operator(
        self,
        cid: int,
        *,
        out: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]:
        r"""Return the multi-level Bézier extraction :math:`C^\epsilon` on cell ``cid``.

        :math:`C^\epsilon = M^\epsilon E^\epsilon` (shape ``(K, n)``) maps the ``target``
        reference basis on the cell to the active hierarchical functions
        (``H^\epsilon = C^\epsilon B``).  For ``target="bezier"``, ``B`` is the Bernstein
        basis on :math:`[0, 1]^d`.  Rows follow :meth:`active_basis`.

        Args:
            cid (int): Active cell flat id in ``[0, num_elements)``.
            out (npt.NDArray[np.float64] | None): Optional output array of shape
                ``(K, n)`` where ``K = active_basis(cid).size`` and
                ``n = (p + 1) ** d``.  Allocated when ``None``.

        Returns:
            npt.NDArray[np.float64]: The operator :math:`C^\epsilon`.

        Raises:
            IndexError: If ``cid`` is out of range.
            ValueError: If ``out`` has the wrong shape, dtype, or is not writeable.
            RuntimeError: If the grid has been modified since construction.
        """
        return self._operator(cid, out, "extraction_operator")

    def __repr__(self) -> str:
        """Return a compact string representation.

        Returns:
            str: Shows dimension, target, and element count.
        """
        return (
            f"MultiLevelExtraction(dim={self.dim}, target={self._target!r}, "
            f"num_elements={self.num_elements})"
        )


def _rebuild_extraction(
    cls: type[MultiLevelExtraction], space: THBSplineSpace, target: Target
) -> MultiLevelExtraction:
    """Rebuild a pickled extraction of class ``cls``, bypassing ``cls.__init__``.

    A subclass's constructor may take other arguments; the previous release's pickles did not
    call it either.

    Args:
        cls (type[MultiLevelExtraction]): The class of the pickled extraction.
        space (THBSplineSpace): Its space.
        target (Target): Its target basis tag.

    Returns:
        MultiLevelExtraction: The extraction (its extra attributes are restored by
        ``__setstate__``).
    """
    extraction = cls.__new__(cls)
    MultiLevelExtraction.__init__(extraction, space, target)
    return extraction
