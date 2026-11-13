"""A validated collection of B-spline patches and the interfaces joining them."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._ext import ext
from ._interface import _from_tuples, _wire

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..bspline import Bspline
    from ._interface import Interface


class MultiPatch:
    """A set of B-spline patches together with their conforming interfaces.

    The patches must agree on everything that would make joining them meaningless
    otherwise: parametric dimension, physical rank, dtype, and whether they are
    rational. Interfaces are detected on construction unless supplied, and supplied
    ones are verified rather than trusted.

    Attributes:
        _patches (tuple[Bspline, ...]): The patches, in the order given.
        _interfaces (tuple[Interface, ...]): The interfaces joining them.
    """

    _patches: tuple[Bspline, ...]
    _interfaces: tuple[Interface, ...]

    def __init__(
        self,
        patches: Sequence[Bspline],
        interfaces: tuple[Interface, ...] | None = None,
        *,
        tol: float | None = None,
    ) -> None:
        """Validate the patches and establish the interfaces.

        Args:
            patches (Sequence[Bspline]): At least one patch. All must share
                parametric dimension, physical rank, dtype and ``is_rational``.
            interfaces (tuple[Interface, ...] | None): Interfaces to use. Each is
                verified against the geometry. Detected automatically when ``None``.
            tol (float | None): Relative tolerance override for detection and
                verification. Defaults to the larger of each pair's own tolerances.

        Raises:
            ValueError: If no patches are given, the patches disagree on dimension,
                rank, dtype or rationality, an interface refers to a patch that does
                not exist, or a supplied interface does not hold for the geometry.
        """
        if not patches:
            raise ValueError("A MultiPatch needs at least one patch")

        # The dtype is the one rule the compiled collection cannot state: it holds one dtype.
        first = patches[0]
        for index, patch in enumerate(patches[1:], start=1):
            if patch.dtype != first.dtype:
                raise ValueError(
                    f"All patches must share dtype; patch 0 has {first.dtype} "
                    f"but patch {index} has {patch.dtype}"
                )

        # The remaining checks (dimension, rank, rationality, the supplied interfaces and their
        # verification, the detection) are the compiled collection's, with its messages.
        found = ext.multipatch.multi_patch_interfaces(
            [patch._h for patch in patches],
            None if interfaces is None else [_wire(i) for i in interfaces],
            tol,
        )
        self._patches = tuple(patches)
        if interfaces is not None:
            self._interfaces = tuple(interfaces)
            return
        self._interfaces = _from_tuples(found)

    @property
    def patches(self) -> tuple[Bspline, ...]:
        """Get the patches.

        Returns:
            tuple[Bspline, ...]: The patches, in the order given at construction.
        """
        return self._patches

    @property
    def interfaces(self) -> tuple[Interface, ...]:
        """Get the interfaces joining the patches.

        Returns:
            tuple[Interface, ...]: The supplied or detected interfaces.
        """
        return self._interfaces

    @property
    def dim(self) -> int:
        """Get the shared parametric dimension of the patches.

        Returns:
            int: Parametric dimension.
        """
        return self._patches[0].dim

    def __len__(self) -> int:
        """Return the number of patches.

        Returns:
            int: Patch count.
        """
        return len(self._patches)
