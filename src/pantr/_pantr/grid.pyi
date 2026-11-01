"""Structured grids, BVH, tags and partitions (private)."""

import enum
from collections.abc import Sequence
from typing import Annotated, overload

import numpy
from numpy.typing import NDArray

import pantr._pantr.geometry
import pantr._pantr.transform

class TensorProductGrid:
    """An immutable tensor-product grid (private)."""

    def __init__(
        self,
        breakpoints: Sequence[
            Annotated[
                NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
            ]
        ],
    ) -> None:
        """Validate and copy the per-axis breakpoints."""

    @property
    def ndim(self) -> int: ...
    @property
    def num_cells(self) -> int: ...
    def cell_bounds(self, cid: int) -> tuple[NDArray[numpy.float64], NDArray[numpy.float64]]:
        """The (lo, hi) corners of a cell, as fresh arrays."""

    def locate(
        self,
        pt: Annotated[
            NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
        ],
    ) -> int | None:
        """The cell containing a point, or None."""

    def locate_many(
        self,
        points: Annotated[
            NDArray[numpy.float64],
            dict(shape=(None, None), order="C", device="cpu", writable=False),
        ],
    ) -> NDArray[numpy.int64]:
        """The cell of each row of (npts, ndim) points; -1 outside."""

    def neighbor_across_facet(self, cid: int, lfid: int) -> int | None:
        """The cell across a local facet, or None."""

    def collect_cell_bounds(self) -> tuple[NDArray[numpy.float64], NDArray[numpy.float64]]:
        """Every cell's (lo, hi) corners as two (num_cells, ndim) arrays."""

    def cell_level(self, cid: int) -> int:
        """The refinement level of a cell."""

    def child_cells(self, cid: int) -> list[int]:
        """The refinement children of a cell."""

    def num_local_facets(self, cid: int) -> int:
        """The number of local facets of a cell."""

    def local_facet_axis_side(self, cid: int, lfid: int) -> tuple[int, int]:
        """(axis, side) of a local facet."""

    def local_facet_bounds(
        self, cid: int, lfid: int
    ) -> tuple[NDArray[numpy.float64], NDArray[numpy.float64]]:
        """The degenerate (lo, hi) box of a facet."""

    def is_mesh_boundary_facet(self, cid: int, lfid: int) -> bool:
        """Whether a facet is on the outer boundary."""

    def neighbors(self, cid: int) -> list[int]:
        """The facet-neighbor cell ids of a cell."""

    def boundary_facets(self) -> NDArray[numpy.int64]:
        """Every outer-boundary facet as (cid, lfid) rows."""

    def hanging_neighbors(self, cid: int, lfid: int) -> list[int]:
        """All active cells sharing a facet."""

    def cell_aabb(self, cid: int) -> pantr._pantr.geometry.AABB:
        """A cell's bounding box."""

    def reference_map(self, cid: int) -> pantr._pantr.transform.AffineTransform:
        """The affine map from the unit cube onto a cell."""

    def cell_bvh(self) -> BVH:
        """The BVH over the cell boxes, built on first use."""

    def bvh_built(self) -> bool:
        """Whether the BVH has been built yet."""

    def query_aabb(self, aabb: pantr._pantr.geometry.AABB) -> NDArray[numpy.int64]:
        """The ids of the cells whose boxes overlap a box."""

    @property
    def cells_per_axis(self) -> list[int]: ...
    @property
    def is_uniform(self) -> bool: ...
    @property
    def bounds(self) -> Annotated[NDArray[numpy.float64], dict(writable=False)]: ...
    def breakpoints(self, axis: int) -> Annotated[NDArray[numpy.float64], dict(writable=False)]:
        """Read-only view of the breakpoints of one axis."""

    def cell_multi_index(self, cid: int) -> list[int]:
        """The per-axis indices of a cell."""

    def flat_cell_index(self, multi: Sequence[int]) -> int:
        """The flat id of a per-axis multi-index."""

    def restrict(
        self,
        ids: Annotated[
            NDArray[numpy.int64], dict(shape=(None,), order="C", device="cpu", writable=False)
        ],
    ) -> tuple[TensorProductGrid, NDArray[numpy.int64], NDArray[numpy.bool_]]:
        """(sub-grid, local_to_global, in_subset) of the bounding-box window."""

    def same_rep(self, other: TensorProductGrid) -> bool:
        """Whether both share one representation."""

class HierarchicalGrid:
    """An immutable hierarchical grid (private)."""

    def __init__(self, root: TensorProductGrid, factor: Sequence[int]) -> None:
        """Every root cell active at level 0 (version 1)."""

    @staticmethod
    def from_blocks(
        root: TensorProductGrid,
        factor: Sequence[int],
        blocks: Sequence[Sequence[tuple[Sequence[int], Sequence[int]]]],
        version: int,
    ) -> HierarchicalGrid:
        """A grid from per-level block lists (pickling and windowing)."""

    @property
    def ndim(self) -> int: ...
    @property
    def num_cells(self) -> int: ...
    def cell_bounds(self, cid: int) -> tuple[NDArray[numpy.float64], NDArray[numpy.float64]]:
        """The (lo, hi) corners of a cell, as fresh arrays."""

    def locate(
        self,
        pt: Annotated[
            NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
        ],
    ) -> int | None:
        """The cell containing a point, or None."""

    def locate_many(
        self,
        points: Annotated[
            NDArray[numpy.float64],
            dict(shape=(None, None), order="C", device="cpu", writable=False),
        ],
    ) -> NDArray[numpy.int64]:
        """The cell of each row of (npts, ndim) points; -1 outside."""

    def neighbor_across_facet(self, cid: int, lfid: int) -> int | None:
        """The cell across a local facet, or None."""

    def collect_cell_bounds(self) -> tuple[NDArray[numpy.float64], NDArray[numpy.float64]]:
        """Every cell's (lo, hi) corners as two (num_cells, ndim) arrays."""

    def cell_level(self, cid: int) -> int:
        """The refinement level of a cell."""

    def child_cells(self, cid: int) -> list[int]:
        """The refinement children of a cell."""

    def num_local_facets(self, cid: int) -> int:
        """The number of local facets of a cell."""

    def local_facet_axis_side(self, cid: int, lfid: int) -> tuple[int, int]:
        """(axis, side) of a local facet."""

    def local_facet_bounds(
        self, cid: int, lfid: int
    ) -> tuple[NDArray[numpy.float64], NDArray[numpy.float64]]:
        """The degenerate (lo, hi) box of a facet."""

    def is_mesh_boundary_facet(self, cid: int, lfid: int) -> bool:
        """Whether a facet is on the outer boundary."""

    def neighbors(self, cid: int) -> list[int]:
        """The facet-neighbor cell ids of a cell."""

    def boundary_facets(self) -> NDArray[numpy.int64]:
        """Every outer-boundary facet as (cid, lfid) rows."""

    def hanging_neighbors(self, cid: int, lfid: int) -> list[int]:
        """All active cells sharing a facet."""

    def cell_aabb(self, cid: int) -> pantr._pantr.geometry.AABB:
        """A cell's bounding box."""

    def reference_map(self, cid: int) -> pantr._pantr.transform.AffineTransform:
        """The affine map from the unit cube onto a cell."""

    def cell_bvh(self) -> BVH:
        """The BVH over the cell boxes, built on first use."""

    def bvh_built(self) -> bool:
        """Whether the BVH has been built yet."""

    def query_aabb(self, aabb: pantr._pantr.geometry.AABB) -> NDArray[numpy.int64]:
        """The ids of the cells whose boxes overlap a box."""

    @property
    def root(self) -> TensorProductGrid: ...
    @property
    def factor(self) -> list[int]: ...
    @property
    def max_level(self) -> int: ...
    @property
    def version(self) -> int: ...
    def cell_multi_index(self, cid: int) -> list[int]:
        """The per-axis index of a cell in its level's coordinates."""

    def cell_id(self, level: int, midx: Sequence[int]) -> int | None:
        """The flat id of an active leaf, or None."""

    def is_active_leaf(self, level: int, midx: Sequence[int]) -> bool:
        """Whether (level, midx) is an active leaf."""

    def level_cells_per_axis(self, level: int) -> list[int]:
        """The cell count per axis at a level."""

    def active_blocks(self, level: int) -> list[tuple[list[int], list[int]]]:
        """The active-leaf blocks at a level."""

    def blocks(self) -> list[list[tuple[list[int], list[int]]]]:
        """Every level's active-leaf blocks."""

    def active_leaf_mask(self, level: int) -> NDArray[numpy.bool_]:
        """The mask of the active leaves at a level."""

    def subdomain_mask(self, level: int) -> NDArray[numpy.bool_]:
        """The mask of a level's refined subdomain."""

    def refined(self, level: int, lo: Sequence[int], hi: Sequence[int]) -> HierarchicalGrid:
        """The grid with a region refined; this grid itself when nothing changes."""

    def refined_cells(self, ids: Sequence[int]) -> HierarchicalGrid:
        """The grid with the given cells refined."""

    def coarsened(self, level: int, lo: Sequence[int], hi: Sequence[int]) -> HierarchicalGrid:
        """The grid with a region coarsened; this grid itself when nothing changes."""

    def coarsened_cells(self, ids: Sequence[int]) -> HierarchicalGrid:
        """The grid with the given cells coarsened."""

    def restrict(
        self,
        ids: Annotated[
            NDArray[numpy.int64], dict(shape=(None,), order="C", device="cpu", writable=False)
        ],
    ) -> tuple[HierarchicalGrid, NDArray[numpy.int64], NDArray[numpy.bool_]]:
        """(sub-grid, local_to_global, in_subset) of the bounding-box window."""

    def export_cells(self) -> tuple[NDArray[numpy.float64], NDArray[numpy.int64]]:
        """The unique nodes (n_nodes, ndim) and the connectivity (num_cells, 2^ndim)."""

    def same_rep(self, other: HierarchicalGrid) -> bool:
        """Whether both share one representation."""

class BVH:
    """An immutable bounding-volume hierarchy (private)."""

    def __init__(
        self,
        node_lo: Annotated[NDArray[numpy.float64], dict(order="C", device="cpu", writable=False)],
        node_hi: Annotated[NDArray[numpy.float64], dict(order="C", device="cpu", writable=False)],
        node_left: Annotated[NDArray[numpy.int64], dict(order="C", device="cpu", writable=False)],
        node_right: Annotated[NDArray[numpy.int64], dict(order="C", device="cpu", writable=False)],
        node_cell: Annotated[NDArray[numpy.int64], dict(order="C", device="cpu", writable=False)],
        n_cells: int,
    ) -> None:
        """Validate and copy a raw tree (node arrays, child pointers, leaf cells)."""

    @staticmethod
    def from_cell_bounds(
        cell_lo: Annotated[NDArray[numpy.float64], dict(order="C", device="cpu", writable=False)],
        cell_hi: Annotated[NDArray[numpy.float64], dict(order="C", device="cpu", writable=False)],
    ) -> BVH:
        """Build the tree over (n_cells, ndim) cell corners."""

    @property
    def ndim(self) -> int: ...
    @property
    def n_cells(self) -> int: ...
    @property
    def n_nodes(self) -> int: ...
    @property
    def node_lo(self) -> Annotated[NDArray[numpy.float64], dict(writable=False)]: ...
    @property
    def node_hi(self) -> Annotated[NDArray[numpy.float64], dict(writable=False)]: ...
    @property
    def node_left(self) -> Annotated[NDArray[numpy.int64], dict(writable=False)]: ...
    @property
    def node_right(self) -> Annotated[NDArray[numpy.int64], dict(writable=False)]: ...
    @property
    def node_cell(self) -> Annotated[NDArray[numpy.int64], dict(writable=False)]: ...
    def query_aabb(self, aabb: pantr._pantr.geometry.AABB) -> NDArray[numpy.int64]:
        """The ids of the cells whose boxes overlap a box."""

class CellTags:
    """An immutable registry of named cell tags (private)."""

    def __init__(self, num_cells: int) -> None:
        """An empty registry."""

    def set(
        self,
        name: str,
        ids: Annotated[
            NDArray[numpy.int64], dict(shape=(None,), order="C", device="cpu", writable=False)
        ],
        values: Annotated[
            NDArray[numpy.int64], dict(shape=(None,), order="C", device="cpu", writable=False)
        ],
    ) -> CellTags:
        """The registry with the tag created or replaced."""

    def get(
        self, name: str
    ) -> tuple[
        Annotated[NDArray[numpy.int64], dict(writable=False)],
        Annotated[NDArray[numpy.int64], dict(writable=False)],
    ]:
        """Read-only (ids, values) views of a tag."""

    @property
    def num_cells(self) -> int: ...
    def remove(self, name: str) -> CellTags:
        """The registry without the tag."""

    def contains(self, name: str) -> bool:
        """Whether a tag of that name exists."""

    def names(self) -> list[str]:
        """The tag names in insertion order."""

    def size(self) -> int:
        """The number of tags."""

class FacetTags:
    """An immutable registry of named facet tags (private)."""

    def __init__(self, num_cells: int, facets_per_cell: int) -> None:
        """An empty registry."""

    @property
    def facets_per_cell(self) -> int: ...
    def set(
        self,
        name: str,
        keys: Annotated[
            NDArray[numpy.int64], dict(shape=(None,), order="C", device="cpu", writable=False)
        ],
        values: Annotated[
            NDArray[numpy.int64], dict(shape=(None,), order="C", device="cpu", writable=False)
        ],
    ) -> FacetTags:
        """The registry with the tag created or replaced; keys are the flattened (M, 2) rows."""

    def get(
        self, name: str
    ) -> tuple[
        Annotated[NDArray[numpy.int64], dict(writable=False)],
        Annotated[NDArray[numpy.int64], dict(writable=False)],
    ]:
        """Read-only (keys (M, 2), values) views of a tag."""

    @property
    def num_cells(self) -> int: ...
    def remove(self, name: str) -> FacetTags:
        """The registry without the tag."""

    def contains(self, name: str) -> bool:
        """Whether a tag of that name exists."""

    def names(self) -> list[str]:
        """The tag names in insertion order."""

    def size(self) -> int:
        """The number of tags."""

class PartitionBackend(enum.Enum):
    """The backends of partition_grid."""

    automatic = 0

    block = 1

    rcb = 2

class Partition:
    """An immutable per-cell owner assignment (private)."""

    def __init__(
        self,
        cell_owner: Annotated[
            NDArray[numpy.int32], dict(shape=(None,), order="C", device="cpu", writable=False)
        ],
        n_parts: int,
    ) -> None:
        """Validate and copy the owners."""

    @property
    def cell_owner(self) -> Annotated[NDArray[numpy.int32], dict(writable=False)]: ...
    @property
    def n_parts(self) -> int: ...
    @property
    def n_cells(self) -> int: ...
    def active_mask(self) -> NDArray[numpy.bool_]:
        """A fresh mask of the cells with an owner."""

    def owned_cells(self, rank: int) -> NDArray[numpy.int64]:
        """The ascending ids of the cells a rank owns."""

def overlay(grid_a: TensorProductGrid, grid_b: TensorProductGrid) -> TensorProductGrid:
    """The coarsest tensor-product grid refining both."""

@overload
def cell_quadrature(
    grid: TensorProductGrid,
    rule_points: Annotated[
        NDArray[numpy.float64], dict(shape=(None, None), order="C", device="cpu", writable=False)
    ],
    rule_weights: Annotated[
        NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
    ],
    cells: Annotated[
        NDArray[numpy.int64], dict(shape=(None,), order="C", device="cpu", writable=False)
    ]
    | None,
) -> tuple[NDArray[numpy.float64], NDArray[numpy.float64]]: ...
@overload
def cell_quadrature(
    grid: HierarchicalGrid,
    rule_points: Annotated[
        NDArray[numpy.float64], dict(shape=(None, None), order="C", device="cpu", writable=False)
    ],
    rule_weights: Annotated[
        NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
    ],
    cells: Annotated[
        NDArray[numpy.int64], dict(shape=(None,), order="C", device="cpu", writable=False)
    ]
    | None,
) -> tuple[NDArray[numpy.float64], NDArray[numpy.float64]]:
    """Map a reference rule onto the cells of a grid: (points, weights)."""

@overload
def partition_grid(
    grid: TensorProductGrid,
    n_parts: int,
    backend: PartitionBackend,
    weights: Annotated[
        NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
    ]
    | None,
    active: Annotated[
        NDArray[numpy.bool_], dict(shape=(None,), order="C", device="cpu", writable=False)
    ]
    | None,
) -> Partition: ...
@overload
def partition_grid(
    grid: HierarchicalGrid,
    n_parts: int,
    backend: PartitionBackend,
    weights: Annotated[
        NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
    ]
    | None,
    active: Annotated[
        NDArray[numpy.bool_], dict(shape=(None,), order="C", device="cpu", writable=False)
    ]
    | None,
) -> Partition:
    """Partition a grid's cells into n_parts rank subdomains."""
