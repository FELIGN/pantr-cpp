"""Public API surface for PaNTr.

Defines package metadata and exported interfaces for polynomial and NURBS
geometric modeling.

The main modules are:
- :mod:`pantr.basis`: 1D polynomial basis evaluation (Bernstein, Lagrange, etc.).
- :mod:`pantr.bspline`: B-spline spaces, geometric objects, and knot vector factories.
- :mod:`pantr.change_basis`: Transformation matrices between different bases.
- :mod:`pantr.geometry`: Axis-aligned bounding boxes and geometry primitives.
- :mod:`pantr.grid`: Structured cell grids (tensor-product, BVH, tags).
- :mod:`pantr.quad`: Quadrature rules and evaluation grid helpers.
- :mod:`pantr.tolerance`: Uniform floating-point tolerance utilities.
- :mod:`pantr.transform`: Affine transformations for geometric objects.
- :mod:`pantr.cad`: Constructive geometry for B-spline curves, surfaces, and volumes.
- :mod:`pantr.bezier`: Bézier curves/surfaces and root finding for Bernstein polynomials.
- :mod:`pantr.mpi`: Optional MPI-distributed B-spline / THB-spline spaces (``mpi`` extra).
- :mod:`pantr.viz`: Optional PyVista/VTK visualization of geometries (``viz`` extra).
- :func:`pantr.cmake_dir`: The installed CMake package, for building against pantr's C++ headers.
"""

from typing import Final

# The compiled extension first: a missing, stale or mismatched module fails the import here, with
# a message saying how to rebuild it, rather than at the first call that reaches it.
from . import _ext  # noqa: F401  # isort: skip
from . import (
    basis,
    bezier,
    bspline,
    cad,
    change_basis,
    geometry,
    grid,
    quad,
    tolerance,
    transform,
)
from ._cmake import cmake_dir
from ._parallel import get_num_threads, num_threads, set_num_threads
from .basis import _basis_utils  # noqa: F401

# Package metadata
__version__: Final[str] = "0.7.0"
__license__: Final[str] = "MIT"
__author__: Final[str] = "Pablo Antolin <pablo.antolin@epfl.ch>"

__all__ = [
    "__author__",
    "__license__",
    "__version__",
    "basis",
    "bezier",
    "bspline",
    "cad",
    "change_basis",
    "cmake_dir",
    "geometry",
    "get_num_threads",
    "grid",
    "num_threads",
    "quad",
    "set_num_threads",
    "tolerance",
    "transform",
]
