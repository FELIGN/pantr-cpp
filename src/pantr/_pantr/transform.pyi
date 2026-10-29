"""Affine maps (private)."""

from typing import Annotated

import numpy
from numpy.typing import NDArray

class AffineTransform:
    """An immutable affine map x -> A x + b."""

    def __init__(
        self,
        matrix: Annotated[NDArray[numpy.float64], dict(order="C", device="cpu", writable=False)],
        translation: Annotated[
            NDArray[numpy.float64], dict(order="C", device="cpu", writable=False)
        ]
        | None,
    ) -> None:
        """Validate and copy A and b."""

    @staticmethod
    def identity(n: int) -> AffineTransform:
        """The identity on R^n."""

    @staticmethod
    def translation(
        offset: Annotated[
            NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
        ],
    ) -> AffineTransform:
        """X -> x + offset."""

    @staticmethod
    def scaling(
        factors: Annotated[
            NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
        ],
        center: Annotated[
            NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
        ]
        | None,
    ) -> AffineTransform:
        """X -> diag(factors) x about a center."""

    @staticmethod
    def rotation_2d(
        angle: float,
        center: Annotated[
            NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
        ]
        | None,
    ) -> AffineTransform:
        """A rotation of R^2 about a center."""

    @staticmethod
    def rotation_3d(
        angle: float,
        axis: Annotated[
            NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
        ],
        center: Annotated[
            NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
        ]
        | None,
    ) -> AffineTransform:
        """A rotation of R^3 about an axis through a center."""

    @staticmethod
    def mirror(
        normal: Annotated[
            NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
        ],
        center: Annotated[
            NDArray[numpy.float64], dict(shape=(None,), order="C", device="cpu", writable=False)
        ]
        | None,
    ) -> AffineTransform:
        """A reflection through a hyperplane."""

    @staticmethod
    def shear(dim: int, component: int, direction: int, factor: float) -> AffineTransform:
        """x[component] += factor x[direction]."""

    @property
    def dim(self) -> int:
        """The dimension n."""

    @property
    def matrix(self) -> Annotated[NDArray[numpy.float64], dict(writable=False)]:
        """Read-only (n, n) matrix A."""

    @property
    def offset(self) -> Annotated[NDArray[numpy.float64], dict(writable=False)]:
        """Read-only (n,) translation b."""

    def inverse(self) -> AffineTransform:
        """The inverse map."""

    def compose(self, other: AffineTransform) -> AffineTransform:
        """Self o other."""

    def apply(
        self,
        points: Annotated[NDArray[numpy.float64], dict(order="C", device="cpu", writable=False)],
    ) -> NDArray[numpy.float64]:
        """A p + b per point, (m, n)."""
