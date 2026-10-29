"""Axis-aligned bounding boxes (private)."""

from typing import Annotated

import numpy
from numpy.typing import NDArray

class AABB:
    """An immutable axis-aligned bounding box."""

    def __init__(
        self,
        lo: Annotated[
            NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
        ],
        hi: Annotated[
            NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
        ],
    ) -> None:
        """Validate and copy the corners."""

    @staticmethod
    def unbounded(ndim: int) -> AABB:
        """The box [-inf, inf]^ndim."""

    @staticmethod
    def empty(ndim: int) -> AABB:
        """The canonical empty box."""

    @staticmethod
    def from_bounds(
        bounds: Annotated[NDArray[numpy.float64], dict(order="C", device="cpu", writable=False)],
    ) -> AABB:
        """A box from an (ndim, 2) array of [lo, hi] rows."""

    @property
    def ndim(self) -> int:
        """The number of axes."""

    @property
    def lo(self) -> Annotated[NDArray[numpy.float64], dict(writable=False)]:
        """Read-only lower corner."""

    @property
    def hi(self) -> Annotated[NDArray[numpy.float64], dict(writable=False)]:
        """Read-only upper corner."""

    def is_empty(self) -> bool:
        """Whether lo > hi on some axis."""

    def contains_point(
        self,
        x: Annotated[
            NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
        ],
    ) -> bool:
        """Whether the closed box contains x."""

    def overlaps(self, other: AABB) -> bool:
        """Whether the closed boxes intersect."""

    def union_with(self, other: AABB) -> AABB:
        """The smallest box containing both (an empty operand is returned as it is)."""

    def intersect(self, other: AABB) -> AABB | None:
        """The intersection, or None."""

    def pad(
        self,
        r: Annotated[
            NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
        ],
    ) -> AABB:
        """Grow each axis by r[i] on both sides."""

    def pad_scalar(self, r: float) -> AABB:
        """Grow every axis by r on both sides."""

    def transform(
        self,
        affine_dim: int,
        matrix: Annotated[NDArray[numpy.float64], dict(order="C", device="cpu", writable=False)],
        offset: Annotated[
            NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
        ],
    ) -> AABB:
        """The bounding box of the image under x -> A x + b."""

    def as_bounds(self) -> NDArray[numpy.float64]:
        """A fresh (ndim, 2) array of [lo, hi] rows."""

    def equals(self, other: AABB) -> bool:
        """IEEE value equality of the corners."""

    def same_rep(self, other: AABB) -> bool:
        """Whether both share one representation."""
