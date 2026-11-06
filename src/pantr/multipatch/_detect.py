"""Detection of conforming interfaces among a set of B-spline patches.

Defines :func:`detect_interfaces`, which finds every pair of patch faces that
coincide geometrically *and* carry compatible tangential knot vectors, together
with the axis correspondence relating them.

Only conforming interfaces are recognized. Two faces that occupy the same region
of space but disagree in degree or interior knots are rejected rather than
approximated: accepting a nested refinement on one side needs a projection the
caller has to opt into, and silently pairing them would produce control-point
matches that are simply wrong.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._ext import ext
from ._interface import Interface, _from_tuples

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..bspline import Bspline


def detect_interfaces(
    patches: Sequence[Bspline], *, tol: float | None = None
) -> tuple[Interface, ...]:
    """Find the conforming interfaces among a set of patches.

    Every pair of faces in canonical order is considered, including two faces of the
    same patch, which is how a closed ring is found. A pair becomes an interface
    only when both conditions hold: all matched control points coincide, and the
    matched tangential knot vectors agree after affine normalization to ``[0, 1]``.
    A face pair that occupies the same space but disagrees in degree or interior
    knots is *not* an interface here.

    Args:
        patches (Sequence[Bspline]): The patches to inspect. All must share the same
            parametric dimension.
        tol (float | None): Dimensionless relative tolerance override. Defaults, per
            pair, to the strict preset of the two dtypes. For coordinates it is scaled
            by the larger of the joint bounding-box diagonal and the largest coordinate
            magnitude. For knots, normalized to ``[0, 1]``, each matched direction of
            the two patches adds its space's knot tolerance over its knot span, so a
            knot vector far from the origin relative to its span is compared at the
            resolution its knots carry: a normalized knot moved by less than the sum
            of the two (each ``8 eps`` times its patch's knot magnitude over its
            span) is not seen.

    Returns:
        tuple[Interface, ...]: The interfaces found, sorted by
        ``(patch_a, face_a, patch_b, face_b)``.

    Raises:
        ValueError: If the patches do not all share the same parametric dimension, or a
            patch has fewer control-point coordinates per point than the rank of an
            earlier one (patches of different ranks are otherwise tolerated, and never
            share an interface).

    Example:
        >>> import numpy as np
        >>> from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D
        >>> space = BsplineSpace([BsplineSpace1D([0, 0, 1, 1], 1)] * 2)
        >>> left = Bspline(space, np.array([[0.0, 0], [0, 1], [1, 0], [1, 1]]))
        >>> right = Bspline(space, np.array([[1.0, 0], [1, 1], [2, 0], [2, 1]]))
        >>> detect_interfaces([left, right])
        (Interface(patch_a=0, face_a=1, patch_b=1, face_b=0, axis_map=(1,), flips=(False,)),)
    """
    if not patches:
        return ()
    found = ext.multipatch.detect_interfaces([patch._h for patch in patches], tol)
    return _from_tuples(found)
