"""Join two B-spline patches along a parametric axis.

Provides :func:`join` for C0-concatenation of B-splines.
"""

from __future__ import annotations

from .._ext import ext
from ..bspline import Bspline
from ..bspline._bspline import _index


def join(bspline1: Bspline, bspline2: Bspline, axis: int) -> Bspline:
    """Join two B-splines along a parametric axis with C0 continuity.

    The two inputs are made compatible on all non-join axes (degree and
    knots), elevated to the same degree on the join axis, and then
    concatenated.  The shared boundary control points are averaged.

    Optionally, knots are removed at the junction to recover higher
    smoothness when the geometry permits it.

    **The two inputs must already meet.**  The face where they join is not independent
    data: ``bspline1``'s boundary there and ``bspline2``'s are two readings of the same
    thing, and averaging them is exact only because they agree.  Pieces that do not meet
    are refused rather than reconciled, since the average of two disagreeing rows lies on
    neither input and the result would then pass through neither, missing each by half
    the gap.  For rational inputs -- including a rational joined to a polynomial, which is
    promoted first -- agreement is required of the **homogeneous** control points, weights
    included, so two boundaries that describe the same points of space under different
    weights are refused too.

    Args:
        bspline1: First B-spline (left side of the join).
        bspline2: Second B-spline (right side of the join).
        axis: Parametric axis along which to join (0-indexed).

    Returns:
        Bspline: A single B-spline spanning both inputs.

    Raises:
        ValueError: If the inputs have different parametric dimensions.
        ValueError: If *axis* is out of range.
        ValueError: If the two inputs disagree anywhere on the face they are joined
            along, by more than ``4096 * eps`` times the largest absolute control-point
            value over the two boundary rows compared. This is
            :func:`create_coons_surface`'s tier, justified in ``verify_corners_2d`` of
            ``include/pantr/cad/coons.hpp``, with two differences that follow from what is
            graded here and are derived in ``verify_shared_boundary`` of
            ``include/pantr/cad/join.hpp``: ``eps`` is that of the coarser of the two inputs'
            **own precisions** rather than always float64, so a float32 seam is graded at
            float32; and the scale is taken over the **seam** rather than over the whole
            boundary skeleton, so a seam must agree to within the tier times its own
            magnitude, not the model's. For a rational join the homogeneous coordinates
            and the weights are graded against their own magnitudes separately, since they
            do not share a physical dimension.
    """
    return Bspline._from_handle(ext.cad.join(bspline1._h, bspline2._h, _index(axis)))
