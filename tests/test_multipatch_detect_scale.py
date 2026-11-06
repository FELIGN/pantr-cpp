"""Interface detection at knot offsets and coordinate magnitudes where its tolerances bite.

Two inherited misses, each a silent "no interface":

- **Knot offset.** The knots are compared after mapping each vector affinely onto ``[0, 1]``, whose
  rounding grows with the knots' magnitude over their span (about ``eps * offset / span``). The
  parametric tolerance was the bare relative one (``4 eps``), so two patches sharing an edge
  exactly, one of whose knot vectors is shifted by 64 or more, were not matched.
- **Corner distance overflow.** The corner pre-filter measured distances as ``sqrt(sum d^2)``,
  whose square overflows to ``inf`` once a coordinate difference reaches about ``1.3e154``, so a
  face pair whose corners agree only to rounding was not matched on patches that large.

Oracle: constructed geometry, two patches sharing the edge ``x = 1`` exactly.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D
from pantr.multipatch import detect_interfaces

_KV = np.array([0.0, 0.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0, 1.0, 1.0])


def _patch(
    x0: float, knots_v: npt.NDArray[np.float64], scale: float = 1.0, nudge_left: bool = False
) -> Bspline:
    """Build a degree-(1, 2) patch on ``[x0, x0 + 1] x [0, 1]``, its coordinates scaled.

    Args:
        x0 (float): Left edge.
        knots_v (npt.NDArray[np.float64]): The second direction's knots.
        scale (float): Factor on every coordinate.
        nudge_left (bool): Move the left edge's ``x`` up by one unit in the last place, as a
            coordinate computed by another route would be.

    Returns:
        Bspline: The patch.
    """
    su = BsplineSpace1D(np.array([0.0, 0.0, 1.0, 1.0]), 1)
    sv = BsplineSpace1D(knots_v, 2)
    ys = np.linspace(0.0, 1.0, sv.num_basis)
    cp = np.stack(np.meshgrid([x0, x0 + 1.0], ys, indexing="ij"), axis=-1) * scale
    if nudge_left:
        cp[0, :, 0] = np.nextafter(cp[0, :, 0], np.inf)
    return Bspline(BsplineSpace([su, sv]), cp)


@pytest.mark.parametrize("offset", [64.0, 1e3, 1e6, 1e9, 1e12], ids=str)
def test_shared_edge_is_found_with_offset_knots(offset: float) -> None:
    """The second patch's knots shifted by ``offset`` describe the same interface."""
    found = detect_interfaces([_patch(0.0, _KV), _patch(1.0, _KV + offset)])
    assert len(found) == 1


def test_moved_interior_knot_is_still_refused_at_an_offset() -> None:
    """Control: at offset ``1e3`` an interior knot moved by ``1e-6`` of the span is not a match.

    The documented resolution at that offset is about ``8 eps * 1e3``, far below ``1e-6``.
    """
    moved = _KV + 1e3
    moved[3] += 1e-6
    assert detect_interfaces([_patch(0.0, _KV), _patch(1.0, moved)]) == ()


@pytest.mark.parametrize("scale", [1.0, 1e150, 1e200, 1e300], ids=str)
def test_shared_edge_is_found_on_huge_patches(scale: float) -> None:
    """A shared edge one ulp apart is found where that ulp's square overflows (from ``1.3e154``)."""
    found = detect_interfaces([_patch(0.0, _KV, scale), _patch(1.0, _KV, scale, nudge_left=True)])
    assert len(found) == 1
