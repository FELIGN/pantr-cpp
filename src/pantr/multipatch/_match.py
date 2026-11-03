"""Control-point index arithmetic across a pair of patch faces.

Defines :func:`match_face_cps`, which pairs the flat control-point ids on two
faces under a given axis correspondence. Pure index arithmetic: no geometry is
read, so it says nothing about whether the two faces actually coincide in space.
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from .._ext import ext

if TYPE_CHECKING:
    from ..bspline import BsplineSpace


def match_face_cps(  # noqa: PLR0913
    space_a: BsplineSpace,
    face_a: int,
    space_b: BsplineSpace,
    face_b: int,
    axis_map: tuple[int, ...],
    flips: tuple[bool, ...],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Pair the flat control-point ids of two faces under an axis correspondence.

    Flat ids are C-order over :attr:`~pantr.bspline.BsplineSpace.num_basis`. The
    ``k``-th tangential axis of ``face_a``, in increasing-axis order, is taken to
    correspond to patch axis ``axis_map[k]`` of ``space_b``, running backwards
    when ``flips[k]``.

    This is index arithmetic only. It does not check that the two faces coincide
    geometrically; :func:`~pantr.multipatch.detect_interfaces` is what establishes
    that.

    Args:
        space_a (BsplineSpace): First patch space.
        face_a (int): Face id on ``space_a``.
        space_b (BsplineSpace): Second patch space.
        face_b (int): Face id on ``space_b``.
        axis_map (tuple[int, ...]): Patch-``b`` axis per tangential axis of
            ``face_a``; a permutation of ``face_b``'s tangential axes.
        flips (tuple[bool, ...]): Whether each of those axis pairs runs opposed.

    Returns:
        tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]: ``(cps_a, cps_b)``,
        equal-length read-only arrays of flat ids. ``cps_a`` is sorted ascending
        and ``cps_b[i]`` is the id matched to ``cps_a[i]``.

    Raises:
        ValueError: If a face id is out of range, the two spaces differ in
            dimension, ``axis_map`` is not a permutation of ``face_b``'s
            tangential axes, ``flips`` has a different length, or the matched
            tangential directions hold different numbers of basis functions.
        TypeError: If a face id or an axis is not an integer, or does not fit in a signed
            64-bit integer.

    Example:
        >>> import numpy as np
        >>> from pantr.bspline import BsplineSpace, BsplineSpace1D
        >>> space = BsplineSpace([BsplineSpace1D([0, 0, 1, 1], 1)] * 2)
        >>> a, b = match_face_cps(space, 1, space, 0, (1,), (False,))
        >>> a.tolist(), b.tolist()
        ([2, 3], [0, 1])
    """
    cps_a, cps_b = ext.multipatch.match_face_cps(
        space_a.num_basis,
        operator.index(face_a),
        space_b.num_basis,
        operator.index(face_b),
        [operator.index(a) for a in axis_map],
        [bool(f) for f in flips],
    )
    cps_a.flags.writeable = False
    cps_b.flags.writeable = False
    return cps_a, cps_b
