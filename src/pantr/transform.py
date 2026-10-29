"""Affine transformations for geometric objects.

Provides the :class:`AffineTransform` class, which represents an affine map
``T(x) = A @ x + b`` in *n*-dimensional space, together with factory methods
for common transformations (translation, rotation, scaling, mirroring, shear)
and composition operators.

Main exports:

- :class:`AffineTransform` — logically immutable affine-transformation object.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np
from numpy import typing as npt

from ._ext import ext

__all__ = ["AffineTransform"]

_cpp: Any = ext.transform
"""The compiled AffineTransform (private; untyped)."""


class AffineTransform:
    """An affine transformation T(x) = A x + b in n-dimensional space.

    The transformation is defined by a square matrix ``A`` (the linear part)
    and a translation vector ``b``.  Instances are treated as immutable: every
    factory method and operator returns a new :class:`AffineTransform`; no
    method mutates an existing instance.

    Attributes:
        _h (Any): The compiled value.
        _matrix (npt.NDArray[np.float64]): The ``(n, n)`` linear part, a read-only view of ``_h``.
        _translation (npt.NDArray[np.float64]): The ``(n,)`` translation, a read-only view of
            ``_h``.
    """

    _h: Any
    _matrix: npt.NDArray[np.float64]
    _translation: npt.NDArray[np.float64]

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        matrix: npt.ArrayLike,
        translation: npt.ArrayLike | None = None,
    ) -> None:
        """Create an affine transformation from a matrix and translation.

        Args:
            matrix (npt.ArrayLike): The ``(n, n)`` linear part of the
                transformation.  Must be a square 2-D array.
            translation (npt.ArrayLike | None): The ``(n,)`` translation
                vector.  If ``None``, defaults to the zero vector.

        Raises:
            ValueError: If *matrix* is not 2-D or not square.
            ValueError: If *translation* length does not match the matrix
                dimension.

        Note:
            Both *matrix* and *translation* are stored as C-contiguous,
            read-only ``float64`` arrays.
        """
        mat = np.ascontiguousarray(np.asarray(matrix, dtype=np.float64))
        tvec = (
            None
            if translation is None
            else np.ascontiguousarray(np.asarray(translation, dtype=np.float64))
        )
        self._set_handle(_cpp.AffineTransform(mat, tvec))

    def _set_handle(self, h: Any) -> None:  # noqa: ANN401
        """Hold a compiled value and cache its read-only views (DR-25: it is immutable).

        Args:
            h (Any): The compiled value.
        """
        self._h = h
        self._matrix = h.matrix
        self._translation = h.offset

    @classmethod
    def _from_handle(cls, h: Any) -> AffineTransform:  # noqa: ANN401
        """Wrap a compiled AffineTransform without copying it.

        Args:
            h (Any): The compiled value.

        Returns:
            AffineTransform: The wrapper.
        """
        t = object.__new__(cls)
        t._set_handle(h)
        return t

    def __reduce__(
        self,
    ) -> tuple[type[AffineTransform], tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
        """Pickle and copy through the constructor.

        Returns:
            tuple[type[AffineTransform], tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
            The class and its matrix and translation.
        """
        return (AffineTransform, (self._matrix, self._translation))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """Get the spatial dimension of the transformation.

        Returns:
            int: Dimension *n* of the transformation.
        """
        return int(self._matrix.shape[0])

    @property
    def matrix(self) -> npt.NDArray[np.float64]:
        """Get the linear part of the transformation.

        Returns:
            npt.NDArray[np.float64]: Read-only ``(n, n)`` matrix.
        """
        return self._matrix

    @property
    def offset(self) -> npt.NDArray[np.float64]:
        """Get the translation (offset) part of the transformation.

        Returns:
            npt.NDArray[np.float64]: Read-only ``(n,)`` vector.
        """
        return self._translation

    @functools.cached_property
    def inverse(self) -> AffineTransform:
        """Get the inverse transformation.

        Computed once and cached; subsequent accesses are free. Safe because
        neither the matrix nor the translation is ever modified after
        construction.

        Returns:
            AffineTransform: The inverse such that ``T @ T.inverse`` is the
            identity.

        Raises:
            ValueError: If the matrix is singular.
        """
        return AffineTransform._from_handle(self._h.inverse())

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @staticmethod
    def identity(n: int) -> AffineTransform:
        """Create the identity transformation in *n* dimensions.

        Args:
            n (int): Spatial dimension.

        Returns:
            AffineTransform: The identity map.
        """
        return AffineTransform._from_handle(_cpp.AffineTransform.identity(n))

    @staticmethod
    def translation(offset: npt.ArrayLike) -> AffineTransform:
        """Create a pure translation.

        Args:
            offset (npt.ArrayLike): Translation vector of length *n*.

        Returns:
            AffineTransform: A translation by *offset*.
        """
        b = np.ascontiguousarray(np.asarray(offset, dtype=np.float64).ravel())
        return AffineTransform._from_handle(_cpp.AffineTransform.translation(b))

    @staticmethod
    def scaling(
        factors: float | npt.ArrayLike,
        *,
        center: npt.ArrayLike | None = None,
    ) -> AffineTransform:
        """Create a scaling transformation.

        Args:
            factors (float | npt.ArrayLike): If a scalar, isotropic scaling
                is applied and *center* (or a separate call) determines the
                dimension.  If an array, anisotropic scaling along each axis.
            center (npt.ArrayLike | None): Optional center point. If given,
                the scaling is performed about this point rather than the
                origin.

        Returns:
            AffineTransform: The scaling transformation.

        Raises:
            ValueError: If *factors* is a scalar and *center* is ``None``
                (dimension cannot be inferred).
            ValueError: If *factors* is a scalar and *center* is not a 1-D
                array-like.
            ValueError: If any factor is non-finite or zero (singular
                transform).
            ValueError: If *factors* is an array and *center* has the wrong
                shape.
        """
        f = np.asarray(factors, dtype=np.float64)
        if f.ndim == 0:
            # Scalar — need center to know dimension.
            if center is None:
                raise ValueError(
                    "An isotropic scaling factor requires a center or an "
                    "array of per-axis factors so the dimension can be "
                    "inferred."
                )
            fval = float(f)
            if not np.isfinite(fval):
                raise ValueError(f"scaling factors must be finite, got {fval!r}.")
            if fval == 0.0:
                raise ValueError(
                    f"scaling factors must be non-zero (singular transform), got {fval!r}."
                )
            c = np.asarray(center, dtype=np.float64)
            if c.ndim != 1:
                raise ValueError(f"center must be a 1-D array, got shape {c.shape}.")
            f = np.full(len(c), fval)
        else:
            f = f.ravel()
            if not np.all(np.isfinite(f)):
                raise ValueError(f"scaling factors must be finite, got {f!r}.")
            if np.any(f == 0.0):
                raise ValueError(
                    f"scaling factors must be non-zero (singular transform), got {f!r}."
                )

        return AffineTransform._from_handle(_cpp.AffineTransform.scaling(f, _center(center)))

    @staticmethod
    def rotation_2d(
        angle: float,
        *,
        center: npt.ArrayLike | None = None,
    ) -> AffineTransform:
        """Create a 2-D counter-clockwise rotation.

        Args:
            angle (float): Rotation angle in radians.
            center (npt.ArrayLike | None): Optional center of rotation.

        Returns:
            AffineTransform: The 2-D rotation.

        Raises:
            ValueError: If *angle* is non-finite.
        """
        return AffineTransform._from_handle(
            _cpp.AffineTransform.rotation_2d(float(angle), _center(center))
        )

    @staticmethod
    def rotation_3d(
        angle: float,
        axis: int | npt.ArrayLike = 2,
        *,
        center: npt.ArrayLike | None = None,
    ) -> AffineTransform:
        """Create a 3-D rotation via the Rodrigues formula.

        Args:
            angle (float): Rotation angle in radians.
            axis (int | npt.ArrayLike): Rotation axis. An ``int`` in
                ``{0, 1, 2}`` selects the corresponding coordinate axis
                (x, y, z).  An array-like of length 3 specifies an arbitrary
                axis (will be normalised internally).
            center (npt.ArrayLike | None): Optional center of rotation.

        Returns:
            AffineTransform: The 3-D rotation.

        Raises:
            ValueError: If *angle* is non-finite.
            ValueError: If an integer axis is not in ``{0, 1, 2}``.
            ValueError: If a vector axis does not have shape ``(3,)``.
            ValueError: If a vector axis is zero or non-finite.
        """
        if isinstance(axis, int | np.integer):
            axis_int = int(axis)
            if axis_int not in (0, 1, 2):
                raise ValueError(f"Integer axis must be 0, 1, or 2, got {axis_int}.")
            u = np.zeros(3, dtype=np.float64)
            u[axis_int] = 1.0
        else:
            u = np.ascontiguousarray(np.asarray(axis, dtype=np.float64).ravel())
            # Checked here as well as in C++ for the message, which spells NumPy's repr of `u`.
            if u.shape == (3,):
                norm = np.linalg.norm(u)
                if norm == 0.0 or not np.isfinite(norm):
                    raise ValueError(f"Rotation axis must be a finite non-zero vector, got {u!r}.")
        return AffineTransform._from_handle(
            _cpp.AffineTransform.rotation_3d(float(angle), u, _center(center))
        )

    @staticmethod
    def mirror(
        normal: npt.ArrayLike,
        *,
        center: npt.ArrayLike | None = None,
    ) -> AffineTransform:
        """Create a reflection (mirror) across a hyperplane.

        The hyperplane passes through the origin (or *center*) and has the
        given *normal* vector.  The Householder formula is used:
        ``A = I - 2 n nᵀ``.

        Args:
            normal (npt.ArrayLike): Normal vector of the mirror plane.  Will
                be normalised internally.
            center (npt.ArrayLike | None): Optional point on the mirror
                plane.

        Returns:
            AffineTransform: The reflection.

        Raises:
            ValueError: If *normal* is zero or non-finite.
        """
        n = np.ascontiguousarray(np.asarray(normal, dtype=np.float64).ravel())
        # Checked here as well as in C++ for the message, which spells NumPy's repr of `n`.
        norm = np.linalg.norm(n)
        if norm == 0.0 or not np.isfinite(norm):
            raise ValueError(f"Mirror normal must be a finite non-zero vector, got {n!r}.")
        return AffineTransform._from_handle(_cpp.AffineTransform.mirror(n, _center(center)))

    @staticmethod
    def shear(
        dim: int,
        component: int,
        direction: int,
        factor: float,
    ) -> AffineTransform:
        """Create a shear transformation.

        The resulting map adds ``factor * x[direction]`` to
        ``x[component]``, leaving all other components unchanged.

        Args:
            dim (int): Spatial dimension.
            component (int): The axis that is modified.
            direction (int): The axis whose value drives the shear.
            factor (float): Shear magnitude.

        Returns:
            AffineTransform: The shear transformation.

        Raises:
            ValueError: If *component* equals *direction*.
            ValueError: If *component* or *direction* is out of range.
            ValueError: If *factor* is non-finite.
        """
        return AffineTransform._from_handle(
            _cpp.AffineTransform.shear(dim, component, direction, factor)
        )

    # ------------------------------------------------------------------
    # Composition and application
    # ------------------------------------------------------------------

    def compose(self, other: AffineTransform) -> AffineTransform:
        """Compose this transformation with *other*.

        Returns the transformation ``self(other(x))``.

        Args:
            other (AffineTransform): The inner transformation.

        Returns:
            AffineTransform: The composed transformation.

        Raises:
            ValueError: If the dimensions do not match.
        """
        return AffineTransform._from_handle(self._h.compose(other._h))

    def __matmul__(self, other: object) -> AffineTransform:
        """Compose via the ``@`` operator.

        Args:
            other (object): Must be an :class:`AffineTransform`.

        Returns:
            AffineTransform: The composed transformation (``self`` after
            ``other``).
        """
        if not isinstance(other, AffineTransform):
            return NotImplemented
        return self.compose(other)

    def __call__(
        self,
        points: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        """Apply the transformation to a set of points.

        Args:
            points (npt.ArrayLike): Points with shape ``(..., n)``.

        Returns:
            npt.NDArray[np.float64]: Transformed points with the same shape.

        Raises:
            ValueError: If the last dimension of *points* does not match
                ``self.dim``.
        """
        pts = np.asarray(points, dtype=np.float64)
        if pts.shape[-1] != self.dim:
            raise ValueError(
                f"Points last dimension ({pts.shape[-1]}) must match "
                f"transform dimension ({self.dim})."
            )
        out: npt.NDArray[np.float64] = self._h.apply(np.ascontiguousarray(pts)).reshape(pts.shape)
        return out

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a developer-friendly string representation.

        Returns:
            str: Representation showing dimension and matrix/translation.
        """
        return (
            f"AffineTransform(dim={self.dim}, "
            f"matrix={self._matrix.tolist()}, "
            f"translation={self._translation.tolist()})"
        )


# ------------------------------------------------------------------
# Module-private helpers
# ------------------------------------------------------------------


def _center(center: npt.ArrayLike | None) -> npt.NDArray[np.float64] | None:
    """Normalise an optional center point for the compiled factories.

    Args:
        center (npt.ArrayLike | None): The center, any shape (ravelled), or ``None``.

    Returns:
        npt.NDArray[np.float64] | None: A C-contiguous 1-D ``float64`` array, or ``None``.
    """
    if center is None:
        return None
    return np.ascontiguousarray(np.asarray(center, dtype=np.float64).ravel())
