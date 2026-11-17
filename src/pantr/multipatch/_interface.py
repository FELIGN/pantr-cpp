"""Interface record between two B-spline patch faces.

Defines :class:`Interface`, the immutable description of one conforming contact
between two patch faces: which faces meet, how their tangential axes correspond,
and which of those axes run in opposite directions.
"""

from __future__ import annotations

import dataclasses
import operator
from typing import TYPE_CHECKING

from .._ext import ext

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclasses.dataclass(frozen=True)
class Interface:
    """A conforming contact between two B-spline patch faces.

    Frozen, so instances are hashable and usable as dictionary keys, and ordered
    canonically: ``(patch_a, face_a)`` is lexicographically smaller than
    ``(patch_b, face_b)``. A patch may meet itself on two different faces, which
    is how a closed ring is expressed.

    ``axis_map`` and ``flips`` describe the reparametrization between the two
    faces. Entry ``k`` refers to the ``k``-th tangential axis of ``face_a`` taken
    in increasing-axis order; ``axis_map[k]`` is the *patch axis of* ``patch_b``
    it corresponds to, and ``flips[k]`` says whether the two run in opposite
    directions.

    Attributes:
        patch_a (int): Index of the first patch.
        face_a (int): Face id on ``patch_a``, as ``2 * axis + side``.
        patch_b (int): Index of the second patch.
        face_b (int): Face id on ``patch_b``.
        axis_map (tuple[int, ...]): Length ``dim - 1``; the patch-``b`` axis
            corresponding to each tangential axis of ``face_a``. A permutation of
            the tangential axes of ``face_b``.
        flips (tuple[bool, ...]): Length ``dim - 1``; ``True`` where that
            tangential axis of ``face_a`` runs against its partner's direction.
    """

    patch_a: int
    face_a: int
    patch_b: int
    face_b: int
    axis_map: tuple[int, ...]
    flips: tuple[bool, ...]

    def __post_init__(self) -> None:
        """Validate the face ids, the axis permutation, and the canonical order.

        Raises:
            ValueError: If the interface joins a face to itself, a face id is out
                of range for the implied dimension, ``axis_map`` is not a
                permutation of ``face_b``'s tangential axes, ``flips`` has a
                different length, or the two sides are not in canonical order.
            TypeError: If an id or an axis is not an integer, or does not fit in a signed
                64-bit integer.
        """
        ext.multipatch.validate_interface(*_wire(self))

    @property
    def dim(self) -> int:
        """Get the parametric dimension of the patches this interface joins.

        Returns:
            int: ``len(axis_map) + 1``.
        """
        return len(self.axis_map) + 1


def _from_tuples(
    found: Iterable[tuple[int, int, int, int, list[int], list[bool]]],
) -> tuple[Interface, ...]:
    """Build interfaces from the compiled library's ``(patch_a, face_a, ...)`` tuples.

    Args:
        found (Iterable[tuple[int, int, int, int, list[int], list[bool]]]): One tuple
            ``(patch_a, face_a, patch_b, face_b, axis_map, flips)`` per interface.

    Returns:
        tuple[Interface, ...]: The interfaces, in the order given.
    """
    return tuple(
        Interface(
            patch_a=patch_a,
            face_a=face_a,
            patch_b=patch_b,
            face_b=face_b,
            axis_map=tuple(axis_map),
            flips=tuple(flips),
        )
        for patch_a, face_a, patch_b, face_b, axis_map, flips in found
    )


def _wire(interface: Interface) -> tuple[int, int, int, int, list[int], list[bool]]:
    """Spell an interface's fields as plain Python ints and bools for the compiled library.

    The fields may hold NumPy integers or booleans (the dataclass does not convert them). The
    compiled signature typed ``bool`` refuses ``np.bool_`` and the ints ``0`` and ``1``, so the
    flips go through ``bool()``; the ids and axes go through ``operator.index``, which accepts any
    integer-like and refuses a float.

    Args:
        interface (Interface): The interface.

    Returns:
        tuple[int, int, int, int, list[int], list[bool]]: ``(patch_a, face_a, patch_b, face_b,
        axis_map, flips)``.
    """
    return (
        operator.index(interface.patch_a),
        operator.index(interface.face_a),
        operator.index(interface.patch_b),
        operator.index(interface.face_b),
        [operator.index(a) for a in interface.axis_map],
        [bool(f) for f in interface.flips],
    )
