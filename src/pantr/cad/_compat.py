"""B-spline compatibility: match degrees and knot vectors across objects.

Provides :func:`make_compat`, which takes N B-splines and returns new objects
that share the same degree and knot vector along each specified axis.
This is the prerequisite for operations like ``create_ruled``,
``create_coons_surface``, and ``join`` that combine control points from
different B-splines.
"""

from __future__ import annotations

from collections.abc import Sequence

from .._ext import ext
from ..bspline import Bspline
from ..bspline._bspline import _index


def make_compat(
    *bsplines: Bspline,
    axes: int | Sequence[int] | None = None,
) -> list[Bspline]:
    """Make B-splines compatible along specified parametric axes.

    Returns new B-splines that share the same polynomial degree and
    knot vector along the given axes.  The geometric mapping of each
    B-spline is preserved exactly.

    The algorithm proceeds in three stages:

    1. **Same domain** -- remap knot vectors to the common envelope
       ``[min(starts), max(ends)]`` per axis.
    2. **Same degree** -- elevate each B-spline to the maximum degree
       per axis.
    3. **Merge knots** -- insert knots so all B-splines share the
       same interior breakpoints with the maximum multiplicity.

    Periodic B-splines are converted to open form before processing.

    Args:
        *bsplines: One or more B-splines to make compatible.
            All must have the same parametric dimension.
            If fewer than two are provided, returns them unchanged.
        axes: Parametric axes along which to operate.  ``None``
            (default) means all axes.  A single ``int`` or a sequence
            of ``int`` selects specific axes.

    Returns:
        list[Bspline]: New B-splines with identical knot structure
        along the specified axes.

    Raises:
        ValueError: If B-splines have different parametric dimensions.
        ValueError: If any axis index is out of range.
    """
    if axes is None:
        axis_list = None
    elif isinstance(axes, int):
        axis_list = [_index(axes)]
    else:
        axis_list = [_index(a) for a in axes]
    made = ext.cad.make_compat([b._h for b in bsplines], axis_list)
    return [Bspline._from_handle(h) for h in made]
