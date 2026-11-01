"""Tensor-product grid with per-axis breakpoint vectors.

:class:`TensorProductGrid` is the first-class concrete :class:`pantr.grid.Grid`:
a tensor product of axis-aligned boxes, stored as one strictly-increasing
breakpoint array per axis. A cell is addressed by a flat row-major (C-order)
identifier -- the last axis varies fastest, as :func:`numpy.ravel_multi_index` and
:func:`numpy.unravel_index` number them -- which matches
:class:`pantr.bspline.SpanwiseElementExtraction` so a grid and an extraction
operator built on the same :class:`pantr.bspline.BsplineSpace` agree on cell ids.

Footprint
---------

The grid stores only the per-axis breakpoint arrays (total size
``sum_k (cells_per_axis[k] + 1)``) plus a handful of small metadata arrays. It
never materializes per-cell bounds or connectivity -- those are computed on
demand. The only ``O(num_cells)`` structure, the :class:`pantr.grid.BVH` behind
:meth:`~pantr.grid.Grid.query_aabb`, is built lazily on first query, so a grid
used purely for geometry (the common case for a B-spline knot grid) stays
proportional to the breakpoints, not to the cell count.

Construction
------------

- ``TensorProductGrid(breakpoints)`` -- from explicit per-axis breakpoint arrays.
- :func:`uniform_grid` -- a uniform grid on a bounding box with given per-axis
  cell counts.
- :func:`tensor_product_grid` -- the knot-span grid of a
  :class:`pantr.bspline.BsplineSpace` (its per-axis unique in-domain knots).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

import numpy as np

from .._ext import ext
from ._grid import Grid, GridRestriction
from ._grid_utils import _as_float64

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from ..bspline import BsplineSpace

_cpp = ext.grid


class TensorProductGrid(Grid):
    """Tensor-product grid of axis-aligned boxes with per-axis breakpoints.

    Cells are numbered in row-major (C) order over :attr:`cells_per_axis` (last
    axis varies fastest). See the module docstring for the footprint and
    construction notes. Size and geometry metadata are exposed through the
    :attr:`ndim`, :attr:`num_cells`, :attr:`cells_per_axis`, :attr:`breakpoints`,
    and :attr:`bounds` properties.
    """

    __slots__ = (
        "_bounds",
        "_breakpoints",
        "_cells_per_axis",
        "_is_uniform",
        "_ndim",
        "_num_cells",
    )

    _bounds: npt.NDArray[np.float64]
    _breakpoints: tuple[npt.NDArray[np.float64], ...]

    def __init__(self, breakpoints: Sequence[npt.ArrayLike]) -> None:
        """Build a tensor-product grid from per-axis breakpoint vectors.

        Args:
            breakpoints (Sequence[npt.ArrayLike]): One strictly increasing
                ``float64`` array-like per axis, each of length
                ``cells_per_axis[d] + 1 >= 2``.

        Raises:
            ValueError: If ``breakpoints`` is empty, any axis has fewer than two
                entries, or any axis is non-finite or not strictly increasing.
            TypeError: If a breakpoint array cannot be cast to ``float64``.
        """
        super().__init__()
        axes = [
            _as_float64(bp, name=f"breakpoints[{d}]").ravel() for d, bp in enumerate(breakpoints)
        ]
        self._init_from_handle(_cpp.TensorProductGrid(axes))

    def _init_from_handle(self, h: Any) -> None:  # noqa: ANN401
        """Adopt a compiled grid and read its metadata and array views once.

        The metadata of an immutable grid is read when the wrapper is built (not
        on every property access); the breakpoint and bound arrays are read-only
        views of the compiled value.

        Args:
            h (Any): The compiled grid.
        """
        self._h = h
        self._ndim = int(h.ndim)
        self._num_cells = int(h.num_cells)
        self._cells_per_axis = tuple(int(n) for n in h.cells_per_axis)
        self._breakpoints = tuple(h.breakpoints(d) for d in range(self._ndim))
        self._bounds = h.bounds
        self._is_uniform = bool(h.is_uniform)

    @classmethod
    def _from_handle(cls, h: Any) -> Self:  # noqa: ANN401
        """Wrap a compiled grid without copying it.

        Args:
            h (Any): The compiled grid.

        Returns:
            Self: The wrapper.
        """
        grid = object.__new__(cls)
        Grid.__init__(grid)
        grid._init_from_handle(h)
        return grid

    def _state(self) -> tuple[Any, ...]:
        """Return the breakpoints, which rebuild the grid.

        Returns:
            tuple[Any, ...]: The per-axis breakpoint arrays.
        """
        return self._breakpoints

    @classmethod
    def _from_state(cls, state: tuple[Any, ...]) -> Self:
        """Rebuild a grid from its breakpoints.

        Args:
            state (tuple[Any, ...]): The result of :meth:`_state`.

        Returns:
            Self: The rebuilt grid.
        """
        return cls(state)

    def _clone(self) -> Self:
        """Return a new wrapper of the same compiled grid.

        Returns:
            Self: A grid sharing this one's immutable compiled value.
        """
        return type(self)._from_handle(self._h)

    # ------------------------------------------------------------------
    # Read-only attributes
    # ------------------------------------------------------------------

    @property
    def ndim(self) -> int:
        """Get the spatial dimension of the grid.

        Returns:
            int: Number of axes (``>= 1``).
        """
        return self._ndim

    @property
    def num_cells(self) -> int:
        """Get the total number of cells.

        Returns:
            int: Product of the per-axis cell counts.
        """
        return self._num_cells

    @property
    def cells_per_axis(self) -> tuple[int, ...]:
        """Get the per-axis cell counts.

        Returns:
            tuple[int, ...]: Length-``ndim`` tuple of per-axis counts.
        """
        return self._cells_per_axis

    @property
    def breakpoints(self) -> tuple[npt.NDArray[np.float64], ...]:
        """Get the per-axis strictly increasing breakpoint arrays.

        Returns:
            tuple[npt.NDArray[np.float64], ...]: Read-only ``float64`` arrays;
            ``breakpoints[d]`` has length ``cells_per_axis[d] + 1``.
        """
        return self._breakpoints

    @property
    def bounds(self) -> npt.NDArray[np.float64]:
        """Get the per-axis ``[lo, hi]`` extremes.

        Returns:
            npt.NDArray[np.float64]: Read-only ``(ndim, 2)`` array.
        """
        return self._bounds

    @property
    def is_uniform(self) -> bool:
        """Get whether every axis has uniform breakpoint spacing.

        Returns:
            bool: ``True`` iff each axis's spacing is constant to within an
            absolute tolerance (``uniform_spacing_atol`` in the C++ grid).
        """
        return self._is_uniform

    # ------------------------------------------------------------------
    # Index helpers (row-major C-order)
    # ------------------------------------------------------------------

    def cell_multi_index(self, cid: int) -> tuple[int, ...]:
        """Return the per-axis indices ``(i_0, ..., i_{ndim-1})`` of cell ``cid``.

        Args:
            cid (int): Flat cell identifier.

        Returns:
            tuple[int, ...]: Length-``ndim`` per-axis index tuple (C-order).

        Raises:
            IndexError: If ``cid`` is out of range.
        """
        return tuple(self._h.cell_multi_index(int(cid)))

    def flat_cell_index(self, multi: Sequence[int]) -> int:
        """Map per-axis cell indices to the flat cell identifier (C-order).

        Args:
            multi (Sequence[int]): Length-``ndim`` per-axis indices; each entry
                must satisfy ``0 <= i_k < cells_per_axis[k]``.

        Returns:
            int: Flat cell identifier.

        Raises:
            ValueError: If ``len(multi) != ndim``.
            IndexError: If any per-axis index is out of range.
        """
        return int(self._h.flat_cell_index([int(i) for i in multi]))

    # ------------------------------------------------------------------
    # Grid contract overrides
    # ------------------------------------------------------------------

    def cell_bounds(
        self,
        cid: int,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return cell ``cid``'s axis-aligned ``(lo, hi)`` corners.

        Args:
            cid (int): Cell identifier.

        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: Fresh,
            writeable length-``ndim`` ``float64`` arrays.

        Raises:
            IndexError: If ``cid`` is out of range.
        """
        lo, hi = self._h.cell_bounds(int(cid))
        return lo, hi

    def locate(self, pt: npt.ArrayLike) -> int | None:
        """Return the cell containing ``pt``, or ``None`` if ``pt`` is outside.

        A point exactly on an interior breakpoint is assigned to the
        lower-indexed cell sharing that face; a point on the outer boundary is
        assigned to the adjacent boundary cell.  Non-finite coordinates
        (NaN or infinity) are outside every cell.

        Args:
            pt (npt.ArrayLike): Length-``ndim`` point.

        Returns:
            int | None: Containing cell id, or ``None`` when ``pt`` lies outside
            the grid domain.

        Raises:
            ValueError: If ``pt`` does not have length ``ndim``.
        """
        found: int | None = self._h.locate(_as_float64(pt, name="pt").ravel())
        return found

    def locate_many(self, points: npt.ArrayLike) -> npt.NDArray[np.int64]:
        """Locate a batch of points by a per-axis binary search, in parallel over the points.

        Args:
            points (npt.ArrayLike): ``(npts, ndim)`` array-like of points, or a
                single length-``ndim`` point.

        Returns:
            npt.NDArray[np.int64]: Shape ``(npts,)`` cell ids; ``-1`` for points
            outside the grid (including points with NaN or infinite coordinates).

        Raises:
            ValueError: If the trailing axis of ``points`` is not ``ndim``.
        """
        out: npt.NDArray[np.int64] = self._h.locate_many(self._normalize_points(points))
        return out

    def neighbor_across_facet(self, cid: int, lfid: int) -> int | None:
        """Return the cell across local facet ``lfid`` of ``cid``, or ``None``.

        Uses the ``lfid = 2 * axis + side`` encoding and per-axis arithmetic.

        Args:
            cid (int): Cell identifier.
            lfid (int): Local facet identifier in ``[0, 2 * ndim)``.

        Returns:
            int | None: Neighbouring cell id, or ``None`` on a boundary facet.

        Raises:
            IndexError: If ``cid`` or ``lfid`` is out of range.
        """
        neighbor: int | None = self._h.neighbor_across_facet(int(cid), int(lfid))
        return neighbor

    def collect_cell_bounds(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Materialize per-cell ``(lo, hi)`` in C-order from the per-axis breakpoints.

        Returns:
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
            ``(cell_lo, cell_hi)`` of shape ``(num_cells, ndim)`` in cell-id
            order.
        """
        cell_lo, cell_hi = self._h.collect_cell_bounds()
        return cell_lo, cell_hi

    def restrict(self, cell_ids: npt.ArrayLike) -> GridRestriction:
        """Return the multi-index bounding-box sub-grid spanning ``cell_ids``.

        The window is the axis-aligned bounding box of the requested cells in
        multi-index space. Each axis of the sub-grid uses a pure slice of this
        grid's breakpoints, ``breakpoints[d][imin_d : imax_d + 2]`` -- never
        re-based or re-clamped -- so the sub-grid's cells coincide exactly with
        the matching cells of this grid.

        Args:
            cell_ids (npt.ArrayLike): Flat cell identifiers to span; duplicates
                are ignored. Each must satisfy ``0 <= cid < num_cells``.

        Returns:
            GridRestriction: The windowed :class:`TensorProductGrid`, its
            ``local_to_global_cell`` map of shape ``(sub.num_cells,)``, and the
            ``in_subset`` mask flagging requested versus bounding-box-fill cells.

        Raises:
            ValueError: If ``cell_ids`` is empty.
            IndexError: If any cell id is out of range ``[0, num_cells)``.
            TypeError: If ``cell_ids`` is not integer-valued.
        """
        handle, local_to_global, in_subset = self._restrict_handle(cell_ids)
        return GridRestriction(TensorProductGrid._from_handle(handle), local_to_global, in_subset)

    def __repr__(self) -> str:
        """Return a compact representation useful for debugging.

        Returns:
            str: ``"TensorProductGrid(ndim=..., cells_per_axis=...)"``.
        """
        return (
            f"TensorProductGrid(ndim={self._ndim}, cells_per_axis={self._cells_per_axis}, "
            f"uniform={self._is_uniform})"
        )


def uniform_grid(
    bounds: npt.ArrayLike,
    cells: int | Sequence[int],
) -> TensorProductGrid:
    """Build a uniform :class:`TensorProductGrid` on a bounding box.

    Args:
        bounds (npt.ArrayLike): ``(ndim, 2)`` array-like of per-axis
            ``[lo, hi]`` pairs with ``ndim >= 1``.
        cells (int | Sequence[int]): Number of cells per axis. A scalar is
            broadcast to every axis; a length-``ndim`` sequence gives per-axis
            counts. Each count must be ``>= 1``.

    Returns:
        TensorProductGrid: A uniform tensor-product grid.

    Raises:
        ValueError: If ``bounds`` does not have shape ``(ndim, 2)``, any axis has
            ``lo >= hi``, ``cells`` has the wrong length, or any count is ``< 1``.
        TypeError: If ``bounds`` cannot be cast to ``float64``.
    """
    arr = _as_float64(bounds, name="bounds")
    if arr.ndim != 2 or arr.shape[-1] != 2:  # noqa: PLR2004
        raise ValueError(f"uniform_grid: bounds must have shape (ndim, 2); got {arr.shape}.")
    ndim = int(arr.shape[0])
    if ndim < 1:
        raise ValueError(f"uniform_grid: ndim must be >= 1; got {ndim}.")
    if not np.all(arr[:, 0] < arr[:, 1]):
        raise ValueError(
            f"uniform_grid: bounds must satisfy lo < hi on every axis; got {arr.tolist()!r}."
        )
    cells_tuple = (int(cells),) * ndim if isinstance(cells, int) else tuple(int(c) for c in cells)
    if len(cells_tuple) != ndim:
        raise ValueError(
            f"uniform_grid: cells must be a scalar or a length-{ndim} sequence; "
            f"got length {len(cells_tuple)}."
        )
    if any(c < 1 for c in cells_tuple):
        raise ValueError(f"uniform_grid: every cells entry must be >= 1; got {cells_tuple!r}.")
    breakpoints = [
        np.linspace(arr[d, 0], arr[d, 1], cells_tuple[d] + 1, dtype=np.float64) for d in range(ndim)
    ]
    return TensorProductGrid(breakpoints)


def tensor_product_grid(space: BsplineSpace) -> TensorProductGrid:
    """Build the knot-span grid of a :class:`pantr.bspline.BsplineSpace`.

    The grid's per-axis breakpoints are the unique in-domain knots of each
    1-D sub-space, so its cells are exactly the space's knot spans and its cell
    ids match :class:`pantr.bspline.SpanwiseElementExtraction` on the same space.

    Args:
        space (BsplineSpace): A tensor-product B-spline space. Periodic
            directions are rejected: a periodic knot vector does not map cleanly
            to a bounded grid.

    Returns:
        TensorProductGrid: A grid whose cells are the knot spans of ``space``.

    Raises:
        ValueError: If any direction of ``space`` is periodic.
    """
    breakpoints: list[npt.NDArray[np.float64]] = []
    for d, sub in enumerate(space.spaces):
        if sub.periodic:
            raise ValueError(
                f"tensor_product_grid: axis {d} is periodic; periodic B-spline spaces "
                "do not map to a bounded tensor-product grid."
            )
        unique, _ = sub.get_unique_knots_and_multiplicity(in_domain=True)
        breakpoints.append(_as_float64(unique, name=f"space.spaces[{d}] unique knots").ravel())
    return TensorProductGrid(breakpoints)


__all__ = ["TensorProductGrid", "tensor_product_grid", "uniform_grid"]
