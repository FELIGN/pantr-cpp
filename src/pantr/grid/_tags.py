"""Sparse named tags for grid cells and facets.

Two lazy registries attach integer labels to a sparse subset of a grid's
entities, following the model used by ``dolfinx.mesh.MeshTags``: each named tag
is a pair of parallel arrays ``(ids, values)`` (or ``(keys, values)`` for
facets). Entities not listed are untagged. This keeps memory proportional to the
number of *tagged* entities, which suits both whole-grid classification
(``in`` / ``out`` / ``cut``) and boundary-condition marking on a handful of
facets.

- :class:`CellTags` -- named ``(cell_ids, values)`` associations, with
  :meth:`CellTags.to_dense` to scatter a tag into a dense ``(num_cells,)`` array
  when a downstream Numba kernel needs one.
- :class:`FacetTags` -- named ``(keys, values)`` associations where each key is a
  ``(cell_id, local_facet_id)`` pair.

Both registries are created lazily by :class:`pantr.grid.Grid` and stay empty
(zero per-cell footprint) until the first :meth:`set` call. Classification logic
(deciding which cells are inside / outside / cut) is the consumer's
responsibility; these classes only store the result.

Each registry is a wrapper over an immutable compiled registry (``_h``): :meth:`set`
and :meth:`remove` replace the wrapper's compiled value, so the wrapper keeps its
identity while a registry value obtained before the change is never altered. The
``(ids, values)`` arrays :meth:`__getitem__` returns are read-only views of the
compiled value and stay valid after later changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import numpy as np

from .._ext import ext
from ._grid_utils import _previous_release_slots

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy.typing as npt


_cpp = ext.grid

# A facet key is the pair (cell_id, local_facet_id); stored as a (M, 2) array.
_FACET_KEY_WIDTH: Final[int] = 2


def _as_int64_1d(values: npt.ArrayLike, *, name: str) -> npt.NDArray[np.int64]:
    """Coerce ``values`` to a 1-D ``int64`` array.

    Args:
        values (npt.ArrayLike): Integer array-like.
        name (str): Argument name, used in error messages.

    Returns:
        npt.NDArray[np.int64]: A C-contiguous 1-D ``int64`` array.

    Raises:
        TypeError: If ``values`` does not have an integer dtype.
        ValueError: If ``values`` is not 1-D.
    """
    arr = np.asarray(values)
    if arr.dtype.kind not in ("i", "u"):
        raise TypeError(f"{name} must have an integer dtype; got {arr.dtype!r}.")
    arr = np.ascontiguousarray(arr.astype(np.int64, copy=False))
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D; got shape {arr.shape}.")
    return arr


def _broadcast_values(
    values: npt.ArrayLike,
    n: int,
    *,
    name: str,
) -> npt.NDArray[np.int64]:
    """Coerce ``values`` to an ``int64`` array of length ``n`` (scalars broadcast).

    Args:
        values (npt.ArrayLike): Scalar integer or length-``n`` integer
            array-like.
        n (int): Required length.
        name (str): Argument name, used in error messages.

    Returns:
        npt.NDArray[np.int64]: A fresh, writeable length-``n`` ``int64`` array.

    Raises:
        TypeError: If ``values`` does not have an integer dtype.
        ValueError: If ``values`` is array-like with a length other than ``n``.
    """
    arr = np.asarray(values)
    if arr.dtype.kind not in ("i", "u"):
        raise TypeError(f"{name} must have an integer dtype; got {arr.dtype!r}.")
    if arr.ndim == 0:
        return np.full(n, int(arr), dtype=np.int64)
    flat = np.ascontiguousarray(arr.astype(np.int64, copy=False)).ravel()
    if flat.shape[0] != n:
        raise ValueError(f"{name} must be a scalar or have length {n}; got length {flat.shape[0]}.")
    return flat


class CellTags:
    """Sparse named integer tags over a grid's cells.

    Each tag named ``name`` is a pair of parallel ``int64`` arrays
    ``(ids, values)`` sorted by ``ids``; a cell not listed in ``ids`` is
    untagged under ``name``. Distinct tag names are independent. The owning
    grid's cell count is exposed through the :attr:`num_cells` property.
    """

    __slots__ = ("_cache", "_h", "_num_cells")

    #: The compiled registry (private).
    _h: Any
    #: The ``(ids, values)`` tuple :meth:`__getitem__` returned for each name (private).
    _cache: dict[str, tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]

    def __init__(self, num_cells: int) -> None:
        """Create an empty cell-tag registry.

        Args:
            num_cells (int): Number of cells in the owning grid (``>= 0``).

        Raises:
            ValueError: If ``num_cells`` is negative.
        """
        self._h = _cpp.CellTags(int(num_cells))
        self._cache = {}
        self._num_cells = int(self._h.num_cells)

    def _copy(self) -> CellTags:
        """Return a new wrapper over the same compiled registry.

        Returns:
            CellTags: A registry equal to this one that changes independently of it.
        """
        new = object.__new__(CellTags)
        new._h = self._h
        new._cache = dict(self._cache)
        new._num_cells = self._num_cells
        return new

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle and copy through the constructor and the tags in insertion order.

        Returns:
            tuple[Any, ...]: The rebuild function and its arguments.
        """
        entries = [(name, *self[name]) for name in self]
        return (_rebuild_cell_tags, (self._num_cells, entries))

    def __setstate__(self, state: Any) -> None:  # noqa: ANN401
        """Rebuild the registry from a pickle written by a previous release.

        A previous release pickled the cell count and a dict of ``name -> (ids, values)``
        as slots. Pickles written by this release go through :meth:`__reduce__` and never
        reach here.

        Args:
            state (Any): The pickled state: ``(None, {slot: value})`` for the protocols
                that pickle slots, or the bare mapping.
        """
        slots = _previous_release_slots(state)
        CellTags.__init__(self, slots["_num_cells"])
        for name, (ids, values) in slots["_tags"].items():
            self.set(name, ids, values)

    @property
    def num_cells(self) -> int:
        """Get the number of cells in the owning grid.

        Returns:
            int: The cell count; valid cell ids are ``[0, num_cells)``.
        """
        return self._num_cells

    def set(self, name: str, ids: npt.ArrayLike, values: npt.ArrayLike) -> None:
        """Create or replace the tag ``name`` with the association ``ids -> values``.

        Args:
            name (str): Tag name. Replaces any existing tag with the same name.
            ids (npt.ArrayLike): 1-D integer array-like of cell ids; each must
                satisfy ``0 <= id < num_cells`` and be unique.
            values (npt.ArrayLike): Scalar integer (broadcast to every id) or a
                1-D integer array-like of the same length as ``ids``.

        Raises:
            TypeError: If ``ids`` or ``values`` is not integer-typed.
            ValueError: If ``ids`` is not 1-D, contains duplicates, or has an id
                out of range, or if ``values`` has a length other than
                ``len(ids)``.
        """
        id_arr = _as_int64_1d(ids, name="ids")
        val_arr = _broadcast_values(values, id_arr.shape[0], name="values")
        self._h = self._h.set(str(name), id_arr, val_arr)
        self._cache.pop(str(name), None)

    def __getitem__(self, name: str) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """Return the ``(ids, values)`` arrays for tag ``name``.

        Args:
            name (str): Tag name.

        Returns:
            tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]: Read-only
            ``(ids, values)`` arrays sorted by ``ids``.

        Raises:
            KeyError: If no tag named ``name`` exists.
        """
        return _tag_entry(self._h, self._cache, name)

    def __contains__(self, name: object) -> bool:
        """Return whether a tag named ``name`` exists.

        Args:
            name (object): Candidate tag name.

        Returns:
            bool: ``True`` iff ``name`` is a registered tag.
        """
        return isinstance(name, str) and bool(self._h.contains(name))

    def __iter__(self) -> Iterator[str]:
        """Iterate over the registered tag names.

        Returns:
            Iterator[str]: Iterator over tag names in insertion order.
        """
        return iter(self.names)

    def __len__(self) -> int:
        """Return the number of registered tags.

        Returns:
            int: Count of distinct tag names.
        """
        return int(self._h.size())

    @property
    def names(self) -> tuple[str, ...]:
        """Get the registered tag names.

        Returns:
            tuple[str, ...]: Tag names in insertion order.
        """
        return tuple(self._h.names())

    def remove(self, name: str) -> None:
        """Delete the tag ``name``.

        Args:
            name (str): Tag name.

        Raises:
            KeyError: If no tag named ``name`` exists.
        """
        if name not in self:
            raise KeyError(name)
        self._h = self._h.remove(name)
        self._cache.pop(name, None)

    def to_dense(
        self,
        name: str,
        *,
        fill: int = 0,
        dtype: npt.DTypeLike = np.int64,
    ) -> npt.NDArray[Any]:
        r"""Scatter tag ``name`` into a dense ``(num_cells,)`` array.

        Untagged cells receive ``fill``. Useful when a downstream Numba kernel
        wants a per-cell label array rather than the sparse representation.

        Args:
            name (str): Tag name.
            fill (int): Value for untagged cells. Defaults to ``0``.
            dtype (npt.DTypeLike): Output integer dtype. Defaults to
                ``numpy.int64``. Values are stored as ``int64`` internally; if
                ``dtype`` is narrower than ``int64`` and any stored value falls
                outside the dtype's representable range, an ``OverflowError`` is
                raised rather than silently truncating the value.

        Returns:
            npt.NDArray[Any]: Fresh, writeable ``(num_cells,)`` array with
            ``dtype`` as the scalar type.

        Raises:
            KeyError: If no tag named ``name`` exists.
            TypeError: If ``dtype`` is not an integer or unsigned-integer dtype.
            OverflowError: If any stored value cannot be represented exactly in
                ``dtype`` (only raised when ``dtype`` is narrower than ``int64``).
        """
        out_dtype = np.dtype(dtype)
        if out_dtype.kind not in ("i", "u"):
            raise TypeError(f"dtype must be an integer dtype; got {out_dtype!r}.")
        ids, values = self[name]
        if ids.shape[0] > 0 and out_dtype.itemsize < 8:  # noqa: PLR2004
            info = np.iinfo(out_dtype)
            vmin, vmax = int(values.min()), int(values.max())
            if vmin < info.min or vmax > info.max:
                raise OverflowError(
                    f"dtype {out_dtype!r} cannot represent all tag values without "
                    f"truncation; value range [{vmin}, {vmax}] exceeds "
                    f"dtype range [{info.min}, {info.max}]."
                )
        out = np.full(self._num_cells, fill, dtype=out_dtype)
        out[ids] = values
        return out


class FacetTags:
    """Sparse named integer tags over a grid's local facets.

    Each facet is addressed by a ``(cell_id, local_facet_id)`` key, with
    ``local_facet_id`` in ``[0, facets_per_cell)``. Each tag named ``name`` is a
    pair ``(keys, values)`` where ``keys`` is an ``(M, 2)`` ``int64`` array of
    ``(cell_id, local_facet_id)`` rows and ``values`` is a length-``M``
    ``int64`` array, sorted lexicographically by key. The owning grid's cell
    count and per-cell facet count are exposed through the :attr:`num_cells` and
    :attr:`facets_per_cell` properties.
    """

    __slots__ = ("_cache", "_facets_per_cell", "_h", "_num_cells")

    #: The compiled registry (private).
    _h: Any
    #: The ``(ids, values)`` tuple :meth:`__getitem__` returned for each name (private).
    _cache: dict[str, tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]

    def __init__(self, num_cells: int, facets_per_cell: int) -> None:
        """Create an empty facet-tag registry.

        Args:
            num_cells (int): Number of cells in the owning grid (``>= 0``).
            facets_per_cell (int): Number of local facets per cell (``>= 1``).

        Raises:
            ValueError: If ``num_cells`` is negative or ``facets_per_cell`` is
                ``< 1``.
        """
        self._h = _cpp.FacetTags(int(num_cells), int(facets_per_cell))
        self._cache = {}
        self._num_cells = int(self._h.num_cells)
        self._facets_per_cell = int(self._h.facets_per_cell)

    def _copy(self) -> FacetTags:
        """Return a new wrapper over the same compiled registry.

        Returns:
            FacetTags: A registry equal to this one that changes independently of it.
        """
        new = object.__new__(FacetTags)
        new._h = self._h
        new._cache = dict(self._cache)
        new._num_cells = self._num_cells
        new._facets_per_cell = self._facets_per_cell
        return new

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle and copy through the constructor and the tags in insertion order.

        Returns:
            tuple[Any, ...]: The rebuild function and its arguments.
        """
        entries = [(name, *self[name]) for name in self]
        return (_rebuild_facet_tags, (self._num_cells, self._facets_per_cell, entries))

    def __setstate__(self, state: Any) -> None:  # noqa: ANN401
        """Rebuild the registry from a pickle written by a previous release.

        A previous release pickled the cell and facet counts and a dict of
        ``name -> (keys, values)`` as slots. Pickles written by this release go through
        :meth:`__reduce__` and never reach here.

        Args:
            state (Any): The pickled state: ``(None, {slot: value})`` for the protocols
                that pickle slots, or the bare mapping.
        """
        slots = _previous_release_slots(state)
        FacetTags.__init__(self, slots["_num_cells"], slots["_facets_per_cell"])
        for name, (keys, values) in slots["_tags"].items():
            self.set(name, keys, values)

    @property
    def num_cells(self) -> int:
        """Get the number of cells in the owning grid.

        Returns:
            int: The cell count.
        """
        return self._num_cells

    @property
    def facets_per_cell(self) -> int:
        """Get the number of local facets per cell.

        Returns:
            int: ``2 * ndim`` for an axis-aligned box grid.
        """
        return self._facets_per_cell

    def set(self, name: str, keys: npt.ArrayLike, values: npt.ArrayLike) -> None:
        """Create or replace the tag ``name`` with the association ``keys -> values``.

        Args:
            name (str): Tag name. Replaces any existing tag with the same name.
            keys (npt.ArrayLike): ``(M, 2)`` integer array-like of
                ``(cell_id, local_facet_id)`` rows; each row must be unique with
                ``0 <= cell_id < num_cells`` and
                ``0 <= local_facet_id < facets_per_cell``.
            values (npt.ArrayLike): Scalar integer (broadcast to every key) or a
                1-D integer array-like of length ``M``.

        Raises:
            TypeError: If ``keys`` or ``values`` is not integer-typed.
            ValueError: If ``keys`` does not have shape ``(M, 2)``, contains a
                duplicate or out-of-range key, or ``values`` has a length other
                than ``M``.
        """
        key_raw = np.asarray(keys)
        if key_raw.dtype.kind not in ("i", "u"):
            raise TypeError(f"keys must have an integer dtype; got {key_raw.dtype!r}.")
        key_arr = np.ascontiguousarray(key_raw.astype(np.int64, copy=False))
        if key_arr.ndim != _FACET_KEY_WIDTH or key_arr.shape[1] != _FACET_KEY_WIDTH:
            raise ValueError(f"keys must have shape (M, 2); got shape {key_arr.shape}.")
        val_arr = _broadcast_values(values, key_arr.shape[0], name="values")
        self._h = self._h.set(str(name), key_arr.reshape(-1), val_arr)
        self._cache.pop(str(name), None)

    def __getitem__(self, name: str) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """Return the ``(keys, values)`` arrays for tag ``name``.

        Args:
            name (str): Tag name.

        Returns:
            tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]: Read-only
            ``(keys, values)`` where ``keys`` has shape ``(M, 2)`` and is sorted
            lexicographically.

        Raises:
            KeyError: If no tag named ``name`` exists.
        """
        return _tag_entry(self._h, self._cache, name)

    def __contains__(self, name: object) -> bool:
        """Return whether a tag named ``name`` exists.

        Args:
            name (object): Candidate tag name.

        Returns:
            bool: ``True`` iff ``name`` is a registered tag.
        """
        return isinstance(name, str) and bool(self._h.contains(name))

    def __iter__(self) -> Iterator[str]:
        """Iterate over the registered tag names.

        Returns:
            Iterator[str]: Iterator over tag names in insertion order.
        """
        return iter(self.names)

    def __len__(self) -> int:
        """Return the number of registered tags.

        Returns:
            int: Count of distinct tag names.
        """
        return int(self._h.size())

    @property
    def names(self) -> tuple[str, ...]:
        """Get the registered tag names.

        Returns:
            tuple[str, ...]: Tag names in insertion order.
        """
        return tuple(self._h.names())

    def remove(self, name: str) -> None:
        """Delete the tag ``name``.

        Args:
            name (str): Tag name.

        Raises:
            KeyError: If no tag named ``name`` exists.
        """
        if name not in self:
            raise KeyError(name)
        self._h = self._h.remove(name)
        self._cache.pop(name, None)

    def to_dense(
        self,
        name: str,
        *,
        fill: int = 0,
        dtype: npt.DTypeLike = np.int64,
    ) -> npt.NDArray[Any]:
        r"""Scatter tag ``name`` into a dense ``(num_cells, facets_per_cell)`` array.

        Untagged facets receive ``fill``. Useful when a downstream Numba kernel
        wants a per-facet label array rather than the sparse representation.

        Args:
            name (str): Tag name.
            fill (int): Value for untagged facets. Defaults to ``0``.
            dtype (npt.DTypeLike): Output integer dtype. Defaults to
                ``numpy.int64``. Values are stored as ``int64`` internally; if
                ``dtype`` is narrower than ``int64`` and any stored value falls
                outside the dtype's representable range, an ``OverflowError`` is
                raised rather than silently truncating the value.

        Returns:
            npt.NDArray[Any]: Fresh, writeable ``(num_cells, facets_per_cell)``
            array with ``dtype`` as the scalar type.

        Raises:
            KeyError: If no tag named ``name`` exists.
            TypeError: If ``dtype`` is not an integer or unsigned-integer dtype.
            OverflowError: If any stored value cannot be represented exactly in
                ``dtype`` (only raised when ``dtype`` is narrower than ``int64``).
        """
        out_dtype = np.dtype(dtype)
        if out_dtype.kind not in ("i", "u"):
            raise TypeError(f"dtype must be an integer dtype; got {out_dtype!r}.")
        keys, values = self[name]
        if keys.shape[0] > 0 and out_dtype.itemsize < 8:  # noqa: PLR2004
            info = np.iinfo(out_dtype)
            vmin, vmax = int(values.min()), int(values.max())
            if vmin < info.min or vmax > info.max:
                raise OverflowError(
                    f"dtype {out_dtype!r} cannot represent all tag values without "
                    f"truncation; value range [{vmin}, {vmax}] exceeds "
                    f"dtype range [{info.min}, {info.max}]."
                )
        out = np.full((self._num_cells, self._facets_per_cell), fill, dtype=out_dtype)
        if keys.shape[0] > 0:
            out[keys[:, 0], keys[:, 1]] = values
        return out

    def __repr__(self) -> str:
        """Return a concise representation showing the cell/facet counts and tag names.

        Returns:
            str: ``"FacetTags(num_cells=..., facets_per_cell=..., tags=[...])"``
        """
        return (
            f"FacetTags(num_cells={self._num_cells}, "
            f"facets_per_cell={self._facets_per_cell}, "
            f"tags={list(self)!r})"
        )


def _tag_entry(
    h: Any,  # noqa: ANN401
    cache: dict[str, tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]],
    name: object,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Return the ``(ids, values)`` tuple of tag ``name``, the same object on every read.

    The compiled value is immutable, so a tuple of read-only views of it stays valid; it is
    built on the first read and kept until ``set`` or ``remove`` of that name drops it.

    Args:
        h (Any): The compiled registry.
        cache (dict[str, tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]): The wrapper's
            tuples by name.
        name (object): The requested tag name.

    Returns:
        tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]: Read-only ``(ids, values)``.

    Raises:
        KeyError: If no tag named ``name`` exists.
    """
    if isinstance(name, str):
        entry = cache.get(name)
        if entry is not None:
            return entry
        if h.contains(name):
            ids, values = h.get(name)
            cache[name] = (ids, values)
            return cache[name]
    raise KeyError(name)


def _rebuild_cell_tags(
    num_cells: int,
    entries: list[tuple[str, npt.NDArray[np.int64], npt.NDArray[np.int64]]],
) -> CellTags:
    """Unpickle a :class:`CellTags`: an empty registry with its tags set in order.

    Args:
        num_cells (int): Number of cells in the owning grid.
        entries (list[tuple[str, npt.NDArray[np.int64], npt.NDArray[np.int64]]]): The tags as
            ``(name, ids, values)``, in insertion order.

    Returns:
        CellTags: The rebuilt registry.
    """
    tags = CellTags(num_cells)
    for name, ids, values in entries:
        tags.set(name, ids, values)
    return tags


def _rebuild_facet_tags(
    num_cells: int,
    facets_per_cell: int,
    entries: list[tuple[str, npt.NDArray[np.int64], npt.NDArray[np.int64]]],
) -> FacetTags:
    """Unpickle a :class:`FacetTags`: an empty registry with its tags set in order.

    Args:
        num_cells (int): Number of cells in the owning grid.
        facets_per_cell (int): Number of local facets per cell.
        entries (list[tuple[str, npt.NDArray[np.int64], npt.NDArray[np.int64]]]): The tags as
            ``(name, keys, values)``, in insertion order.

    Returns:
        FacetTags: The rebuilt registry.
    """
    tags = FacetTags(num_cells, facets_per_cell)
    for name, keys, values in entries:
        tags.set(name, keys, values)
    return tags


__all__ = ["CellTags", "FacetTags"]
