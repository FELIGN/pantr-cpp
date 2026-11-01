"""Overlay of two tensor-product grids.

:func:`overlay` takes two :class:`TensorProductGrid` instances and returns a
third whose per-axis breakpoints are the union of both inputs' breakpoints,
restricted to the intersection of their domains. The overlay is the coarsest
tensor-product grid that simultaneously refines both inputs: every overlay cell
is contained in exactly one cell of each input.

This is the background-grid bridge for immersed / unfitted quadrature. When a
B-spline (one knot structure) is integrated over an immersion grid (a different
structure), each overlay cell falls inside one polynomial piece of the B-spline
*and* one immersion cell, so a Bernstein patch can be reused without further
subdivision.

The operation is symmetric -- ``overlay(a, b)`` and ``overlay(b, a)`` produce
the same breakpoints up to floating-point noise -- and is defined for any
``ndim >= 1``.
"""

from __future__ import annotations

from .._ext import ext
from ._tensor_product_grid import TensorProductGrid

_cpp = ext.grid


def overlay(grid_a: TensorProductGrid, grid_b: TensorProductGrid) -> TensorProductGrid:
    """Return the tensor-product overlay of ``grid_a`` and ``grid_b``.

    The overlay's per-axis breakpoints are the sorted union of both inputs'
    breakpoint arrays restricted to the intersection of their domains.
    Breakpoints closer than the default *relative* tolerance
    (:func:`pantr.tolerance.get_default`) times the axis's own magnitude -- the
    intersection window and the coordinates bounding it -- are merged into one, so
    the merge behaves the same on ``[0, 1]`` as on ``[1e6, 1e6 + 1]``. The result
    is the coarsest :class:`TensorProductGrid` that refines both inputs.

    Args:
        grid_a (TensorProductGrid): First input grid.
        grid_b (TensorProductGrid): Second input grid; must share
            :attr:`~TensorProductGrid.ndim` with ``grid_a`` and have a non-empty
            domain intersection on every axis.

    Returns:
        TensorProductGrid: The overlay grid.

    Raises:
        TypeError: If either argument is not a :class:`TensorProductGrid`.
        ValueError: If the grids have different
            :attr:`~TensorProductGrid.ndim`, or if their domains do not overlap
            on some axis (the per-axis intersection is empty or degenerate).
        OverflowError: If the merged grid would have 2**63 cells or more.

    Example:
        >>> import numpy.testing as npt
        >>> from pantr.grid import uniform_grid, overlay
        >>> a = uniform_grid([[0.0, 1.0]], 2)
        >>> b = uniform_grid([[0.0, 1.0]], 3)
        >>> npt.assert_allclose(
        ...     overlay(a, b).breakpoints[0],
        ...     [0.0, 1/3, 0.5, 2/3, 1.0],
        ... )
    """
    if not isinstance(grid_a, TensorProductGrid) or not isinstance(grid_b, TensorProductGrid):
        raise TypeError(
            "overlay() requires two TensorProductGrid instances; got "
            f"{type(grid_a).__name__!r} and {type(grid_b).__name__!r}."
        )
    return TensorProductGrid._from_handle(_cpp.overlay(grid_a._h, grid_b._h))
