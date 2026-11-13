"""Every 64-bit integer array a public call returns is typed as NumPy's ``int64``.

NumPy has two 64-bit signed integer types on Linux, ``np.int64`` (``np.long``, char ``'l'``) and
``np.longlong`` (char ``'q'``): they compare equal as dtypes, but ``a.dtype.type is np.int64``,
``isinstance(a[0], np.int64)`` and ``np.issubdtype(a.dtype, np.int64)`` hold only for the first,
and ``repr`` spells the second out. The previous release returned ``np.int64`` everywhere. The
invariant asserted is the type, never the character, because ``np.int64``'s character differs by
platform.

How the routes were enumerated: every binding that hands an int64 array to Python goes through
``to_numpy``, ``copy_to_numpy`` or a read-only view (``bindings/src/arrays.hpp``); each call site
of those with an ``int64`` element type was traced to the public names that expose it (a
``std::vector`` of integers crosses as a list and is not an array). Routes whose wrapper rebuilds
the array in Python (``tabulate_basis`` first indices, ``to_dense``, ``cell_supports``, the
batched Bezier root counts) are included as controls. Every call's whole result is then walked,
tuples, lists, dicts and named-tuple fields included, so an int64 array the census missed inside a
result still fails here.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import pantr.bezier as bz
import pantr.bspline as bs
import pantr.grid as gr
from pantr.geometry import AABB
from pantr.mpi import DistributedSpace


class _FakeComm:
    """The two attributes a ``DistributedSpace`` reads at construction (rank 0 of 2)."""

    rank = 0
    size = 2


def _int64_arrays(value: object, path: str = "") -> list[tuple[str, npt.NDArray[Any]]]:
    """Collect every 64-bit signed integer array inside ``value``, with its path."""
    found: list[tuple[str, npt.NDArray[Any]]] = []
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "i" and value.dtype.itemsize == 8:
            found.append((path, value))
    elif isinstance(value, dict):
        for key, item in value.items():
            found.extend(_int64_arrays(item, f"{path}[{key!r}]"))
    elif isinstance(value, tuple | list):
        fields = getattr(value, "_fields", None)
        for i, item in enumerate(value):
            found.extend(_int64_arrays(item, f"{path}.{fields[i]}" if fields else f"{path}[{i}]"))
    return found


def _grids() -> dict[str, Any]:
    tpg = gr.TensorProductGrid([[0.0, 1.0, 2.0], [0.0, 1.0]])
    hg = gr.hierarchical_grid(gr.uniform_grid([[0.0, 1.0], [0.0, 1.0]], 2), 2)
    hg.refine(0, [0, 0], [1, 1])
    return {"tpg": tpg, "hg": hg}


def _grid_routes() -> dict[str, Callable[[], object]]:
    """The public grid calls, by name, each returning what the caller receives."""
    pts = np.array([[0.5, 0.5], [3.0, 3.0]])
    box = AABB([0.0, 0.0], [1.5, 1.5])
    routes: dict[str, Callable[[], object]] = {}
    for name, g in _grids().items():
        calls: dict[str, Callable[[Any], object]] = {
            "locate_many": lambda g: g.locate_many(pts),
            "boundary_facets": lambda g: g.boundary_facets(),
            "query_aabb": lambda g: g.query_aabb(box),
            "restrict": lambda g: tuple(g.restrict([0, 1])),
            "cell_bvh.query_aabb": lambda g: g.cell_bvh().query_aabb(box),
            "cell_bvh.nodes": lambda g: (
                g.cell_bvh().node_left,
                g.cell_bvh().node_right,
                g.cell_bvh().node_cell,
            ),
        }
        for call, f in calls.items():
            routes[f"{name}.{call}"] = functools.partial(f, g)

        def cell_tag(g: Any = g) -> object:
            g.cell_tags.set("a", [0, 1], 7)
            return (g.cell_tags["a"], g.cell_tags.to_dense("a"))

        def facet_tag(g: Any = g) -> object:
            g.facet_tags.set("f", [[0, 1], [1, 2]], 3)
            return (g.facet_tags["f"], g.facet_tags.to_dense("f"))

        routes[f"{name}.cell_tags"] = cell_tag
        routes[f"{name}.facet_tags"] = facet_tag
    routes["hg.export_cells"] = lambda: _grids()["hg"].export_cells()

    tpg = _grids()["tpg"]
    bvh = gr.BVH.from_cell_bounds(*tpg.collect_cell_bounds())
    routes["BVH.from_cell_bounds.nodes"] = lambda: (bvh.node_left, bvh.node_right, bvh.node_cell)
    routes["BVH.query_aabb"] = lambda: bvh.query_aabb(box)

    def raw_bvh() -> object:
        own = gr.BVH(
            bvh.node_lo,
            bvh.node_hi,
            np.array(bvh.node_left, dtype=np.int64),
            np.array(bvh.node_right, dtype=np.int64),
            np.array(bvh.node_cell, dtype=np.int64),
            n_cells=bvh.n_cells,
        )
        return (own.node_left, own.node_right, own.node_cell)

    routes["BVH(...).nodes"] = raw_bvh
    unit = gr.uniform_grid([[0.0, 1.0], [0.0, 1.0]], 2)
    routes["partition_grid.owned_cells"] = lambda: gr.partition_grid(unit, 2).owned_cells(0)
    routes["Partition.owned_cells"] = lambda: gr.Partition(np.array([0, 1, 0, 1]), 2).owned_cells(1)

    return routes


def _bspline_routes() -> dict[str, Callable[[], object]]:
    """The public B-spline, root and distributed calls, by name."""
    routes: dict[str, Callable[[], object]] = {}
    sp1 = bs.BsplineSpace1D([0, 0, 0, 1, 2, 2, 2], 2)
    routes["BsplineSpace1D.first_basis_per_interval"] = sp1.first_basis_per_interval
    routes["BsplineSpace1D.get_unique_knots_and_multiplicity"] = lambda: (
        sp1.get_unique_knots_and_multiplicity(),
        sp1.get_unique_knots_and_multiplicity(in_domain=True),
    )
    routes["BsplineSpace1D.restrict"] = lambda: sp1.restrict(0, 1)
    routes["BsplineSpace1D.tabulate_basis"] = lambda: (
        sp1.tabulate_basis([0.5]),
        sp1.tabulate_basis_derivatives([0.5], 1),
    )
    sp = bs.create_uniform_space(2, [3, 3])
    routes["BsplineSpace.restrict"] = lambda: tuple(sp.restrict([0, 1]))
    routes["BsplineSpace.tabulate_basis"] = lambda: sp.tabulate_basis(np.array([[0.5, 0.5]]))
    routes["BsplineSpace.cell_supports"] = lambda: sp.cell_supports([0])
    routes["BsplineSpace.spaces[0].first_basis_per_interval"] = lambda: sp.spaces[
        0
    ].first_basis_per_interval()
    ex = bs.SpanwiseElementExtraction(sp, "bezier")
    routes["SpanwiseElementExtraction.idx_maps_1d"] = lambda: ex.idx_maps_1d
    routes["make_struct_view.idx_maps_1d"] = lambda: bs.make_struct_view(ex).idx_maps_1d
    thb = bs.create_thb_space(sp, 2)
    routes["THBSplineSpace.restrict"] = lambda: tuple(thb.restrict([0, 1]))
    part9 = gr.partition_grid(gr.uniform_grid([[0.0, 1.0], [0.0, 1.0]], 3), 2)
    routes["build_local"] = lambda: tuple(bs.build_local(sp, part9, 0))

    def local_thb() -> object:
        local = bs.build_local(thb, gr.partition_grid(thb.grid, 2), 0)
        assert local is not None
        return tuple(local)

    routes["build_local(thb)"] = local_thb

    def distributed() -> object:
        ds = DistributedSpace(sp, part9, _FakeComm())
        local = ds.local
        assert local is not None
        return (ds.owned_cells, tuple(local))

    routes["DistributedSpace"] = distributed
    routes["bezier.find_roots(batch)"] = lambda: bz.find_roots(
        [bz.Bezier([1.0, -1.0]), bz.Bezier([-1.0, 2.0])]
    )

    u, v = np.meshgrid([0.0, 0.5, 1.0], [0.0, 0.5, 1.0], indexing="ij")
    surface = bs.Bspline(bs.create_uniform_space(1, [2, 2]), np.stack([u, v], axis=-1))
    routes["Bspline.locate"] = lambda: surface.locate(np.array([[0.25, 0.75], [5.0, 5.0]]))
    return routes


_ROUTES = {**_grid_routes(), **_bspline_routes()}


@pytest.mark.parametrize("name", sorted(_ROUTES))
def test_int64_outputs_are_numpy_int64(name: str) -> None:
    """Every 64-bit integer array in a public call's result has ``dtype.type is np.int64``."""
    arrays = _int64_arrays(_ROUTES[name]())
    assert arrays, f"{name} returned no int64 array; the route is stale"
    for path, a in arrays:
        assert a.dtype.type is np.int64, f"{name}{path}: {a.dtype.char!r}"
        assert a.dtype == np.dtype(np.int64)
        assert np.issubdtype(a.dtype, np.int64)
        if a.size:
            assert isinstance(a.flat[0], np.int64)
