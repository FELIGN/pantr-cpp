"""Quadrature rules and evaluation grid helpers for 1D integration.

This module provides:

- 1D quadrature rules on ``[0, 1]``: trapezoidal (equispaced), Gauss-Legendre,
  Gauss-Lobatto-Legendre, Chebyshev-Gauss (1st and 2nd kind).
- :class:`PointsLattice`: a multi-dimensional tensor-product evaluation grid.
- :func:`create_lagrange_points_lattice`: factory for Lagrange-node lattices.
- :class:`QuadratureRule`: a d-dimensional quadrature rule on the unit cube,
  with :func:`tensor_product_quadrature` and :func:`gauss_legendre_quadrature`
  factories; the reference rule consumed by :func:`pantr.grid.cell_quadrature`.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, Final, Literal, cast

import numpy as np
import numpy.typing as npt

from ._array_utils import _is_float32
from ._ext import ext

if TYPE_CHECKING:
    from ._pantr.quad import PointsLatticeF32, PointsLatticeF64
    from ._pantr.quad import QuadratureRule as _QuadratureRuleHandle
    from .basis import LagrangeVariant

_Rule1D = tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]]
"""A 1D rule as returned to the caller: ``(nodes, weights)``."""

_RuleFn = Callable[[int], _Rule1D]
"""A C++ rule builder: ``n_pts -> (nodes, weights)``, validating ``n_pts``."""

_RULES: Final[dict[str, tuple[_RuleFn, _RuleFn]]] = {
    name: (getattr(ext.quad, name), getattr(ext.quad, f"{name}_f32"))
    for name in (
        "trapezoidal_1d",
        "gauss_legendre_1d",
        "gauss_lobatto_legendre_1d",
        "chebyshev_gauss_1st_kind_1d",
        "chebyshev_gauss_2nd_kind_1d",
        "tanh_sinh_1d",
    )
}
"""The C++ rule builders by name, as ``(double, float)`` instantiations."""


def _rule(name: str, n_pts: int, dtype: npt.DTypeLike) -> _Rule1D:
    """Build a 1D rule in the C++ library, in the instantiation ``dtype`` selects.

    Args:
        name (str): The rule's name in :data:`_RULES`.
        n_pts (int): Number of points; the C++ builder refuses one below the rule's minimum.
        dtype (npt.DTypeLike): ``float32`` or ``float64``.

    Returns:
        _Rule1D: ``(nodes, weights)`` on ``[0, 1]``.

    Raises:
        ValueError: If ``dtype`` is not float32 or float64, or ``n_pts`` is below the minimum.
    """
    f64, f32 = _RULES[name]
    return f32(n_pts) if _is_float32(dtype) else f64(n_pts)


def get_trapezoidal_1d(
    n_pts: int, dtype: npt.DTypeLike = np.float64
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]]:
    """Get trapezoidal quadrature nodes on [0, 1] for the given number of points.

    If n_pts == 1, the nodes are [0.5] and the weights are [1.0].

    Args:
        n_pts (int): The number of points. Must be at least 1.
        dtype (npt.DTypeLike): The dtype of the nodes. If must be float32 or float64.
            Defaults to float64.

    Returns:
        tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]]:
            The nodes and weights.

    Raises:
        ValueError: If n_pts is less than 1 or dtype is not float32 or float64.
    """
    return _rule("trapezoidal_1d", n_pts, dtype)


def get_gauss_legendre_1d(
    n_pts: int, dtype: npt.DTypeLike = np.float64
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]]:
    """Get Gauss-Legendre quadrature nodes on [0, 1] for the given number of points.

    Args:
        n_pts (int): The number of points. Must be at least 1.
        dtype (npt.DTypeLike): The dtype of the nodes. If must be float32 or float64.
            Defaults to float64.

    Returns:
        tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]]:
            The nodes and weights.

    Raises:
        ValueError: If n_pts is less than 1 or dtype is not float32 or float64.
    """
    return _rule("gauss_legendre_1d", n_pts, dtype)


def get_gauss_lobatto_legendre_1d(
    n_pts: int, dtype: npt.DTypeLike = np.float64
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]]:
    """Get Gauss-Lobatto-Legendre quadrature nodes on [0, 1] for the given number of points.

    Args:
        n_pts (int): The number of points. Must be at least 2.
        dtype (npt.DTypeLike): The dtype of the nodes. If must be float32 or float64.
            Defaults to float64.

    Returns:
        tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]]:
            The nodes and weights.

    Raises:
        ValueError: If n_pts is less than 2 or dtype is not float32 or float64.
    """
    return _rule("gauss_lobatto_legendre_1d", n_pts, dtype)


def get_chebyshev_gauss_1st_kind_1d(
    n_pts: int, dtype: npt.DTypeLike = np.float64
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]]:
    """Get Chebyshev-Gauss quadrature of the first kind on [0, 1] for the given number of points.

    If n_pts == 1, the nodes are [0.5] and the weights are [1.0].

    Args:
        n_pts (int): Number of quadrature points. Must be at least 1.
        dtype (npt.DTypeLike): Floating dtype for nodes/weights; float32 or float64.
            Defaults to float64.

    Returns:
        tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]]:
            The nodes and weights.

    Raises:
        ValueError: If n_pts is less than 1 or dtype is not float32 or float64.
    """
    return _rule("chebyshev_gauss_1st_kind_1d", n_pts, dtype)


def get_modified_chebyshev_nodes_1d(
    n_pts: int, dtype: npt.DTypeLike = np.float64
) -> npt.NDArray[np.float32 | np.float64]:
    """Get modified Chebyshev nodes on [0, 1] for the given number of points.

    Returns Chebyshev nodes of the second kind (Chebyshev-Lobatto points)
    mapped to [0, 1].  These include both endpoints and are suitable for
    polynomial interpolation into the Bernstein basis.

    Unlike the quadrature functions in this module, this returns only nodes
    (no weights) since it is intended for interpolation, not integration.

    Args:
        n_pts (int): Number of nodes.  Must be at least 2.
        dtype (npt.DTypeLike): Floating dtype; float32 or float64.
            Defaults to float64.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Array of shape ``(n_pts,)`` with
        nodes in [0, 1], starting at 0 and ending at 1.

    Raises:
        ValueError: If *n_pts* < 2 or *dtype* is not float32 or float64.
    """
    if _is_float32(dtype):
        return ext.quad.modified_chebyshev_nodes_1d_f32(n_pts)
    return ext.quad.modified_chebyshev_nodes_1d(n_pts)


def get_chebyshev_gauss_2nd_kind_1d(
    n_pts: int, dtype: npt.DTypeLike = np.float64
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]]:
    r"""Get Chebyshev-Gauss quadrature of the second kind on [0, 1] for the given number of points.

    The rule integrates against the Chebyshev second-kind weight function
    mapped to [0, 1]: :math:`\int_0^1 f(x) \sqrt{1 - (2x - 1)^2}\, dx \approx
    \sum_k w_k f(x_k)`, exactly for polynomials of degree up to
    ``2 * n_pts - 1``.  The nodes are the mapped roots of the Chebyshev
    polynomial of the second kind :math:`U_{n_pts}` -- all interior (no
    endpoints).  For endpoint-including Chebyshev-Lobatto *interpolation*
    nodes, use :func:`get_modified_chebyshev_nodes_1d` instead.

    Args:
        n_pts (int): Number of quadrature points. Must be at least 2.
        dtype (npt.DTypeLike): Floating dtype for nodes/weights; float32 or float64.
            Defaults to float64.

    Returns:
        tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]]:
            The nodes (ascending, interior) and weights.

    Raises:
        ValueError: If n_pts is less than 2 or dtype is not float32 or float64.
    """
    return _rule("chebyshev_gauss_2nd_kind_1d", n_pts, dtype)


def get_tanh_sinh_1d(
    n_pts: int, dtype: npt.DTypeLike = np.float64
) -> tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]]:
    """Get tanh-sinh quadrature nodes on [0, 1] for the given number of points.

    Tanh-sinh (double-exponential) quadrature clusters nodes near the
    endpoints of the interval, making it well suited for integrands with
    endpoint singularities or steep boundary layers.  The scheme is
    symmetric, and every returned node lies strictly inside ``(0, 1)``: a node
    that would sit closer to an endpoint than *dtype* can resolve ends the rule
    instead of being moved onto the boundary, so the number of returned nodes
    may be less than *n_pts*, and fewer in ``float32`` than in ``float64``.
    Placing a node *on* the boundary would defeat the purpose of the rule, an
    endpoint singularity being exactly what it is meant to integrate.

    Args:
        n_pts (int): Requested number of quadrature points.  Must be at
            least 1.
        dtype (npt.DTypeLike): Floating-point dtype for the output arrays.
            Must be ``float32`` or ``float64``.  Defaults to ``float64``.

    Returns:
        tuple[npt.NDArray[np.float32 | np.float64], npt.NDArray[np.float32 | np.float64]]:
            ``(nodes, weights)`` on ``[0, 1]``.  Both arrays have the same
            length, which may be less than *n_pts* because the rule is
            truncated where the endpoint gap stops being resolvable.  Weights
            sum to 1 up to the rounding of that sum.

    Raises:
        ValueError: If *n_pts* < 1 or *dtype* is not ``float32``/``float64``.

    Note:
        Truncation puts a floor on what a singular integrand can reach, and
        raising *n_pts* past it buys nothing.  The rule covers
        ``[delta, 1 - delta]`` with ``delta`` of order ``eps / 2``, so the error
        is the integral over the two neglected tails.  For ``x**-0.5`` that is
        ``2 * sqrt(delta) = 2e-8`` (measured 2.0e-8 from *n_pts* 49 on) and for
        ``log(x)`` it is ``delta * (log(delta) - 1) = 4e-15`` (measured 3.9e-15).
        A smooth integrand is unaffected and converges to machine precision.

    Example:
        >>> nodes, weights = get_tanh_sinh_1d(5)
        >>> nodes.shape[0] <= 5
        True
        >>> bool(((nodes > 0.0) & (nodes < 1.0)).all())
        True
        >>> bool(abs(weights.sum() - 1.0) < 1e-14)
        True
    """
    return _rule("tanh_sinh_1d", n_pts, dtype)


def _as_coordinates(pts: npt.ArrayLike) -> npt.NDArray[np.float32 | np.float64]:
    """Turn one direction's coordinates into the array the C++ lattice copies.

    ``float32`` and ``float64`` keep their dtype; any other real dtype (an integer, ``float16``,
    a non-native byte order) is converted to ``float64``, the lattice being a C++ value of one of
    those two types. The dimensionality is preserved, so
    a 0-D or 2-D input is still there for the lattice's validation to reject.

    Args:
        pts (npt.ArrayLike): The caller's coordinates.

    Returns:
        npt.NDArray[np.float32 | np.float64]: A C-contiguous array (a view of the caller's when it
        already is one; the C++ lattice copies it).

    Raises:
        ValueError: If the coordinates are complex.
    """
    arr = np.asarray(pts)
    if np.iscomplexobj(arr):
        raise ValueError("PointsLattice coordinates must be real")
    # Compared as dtypes, not as scalar types, so a non-native byte order is converted too.
    if arr.dtype not in (np.float32, np.float64):
        arr = arr.astype(np.float64)
    return np.ascontiguousarray(arr)


class PointsLattice:
    """A tensor-product grid of evaluation points in multiple dimensions.

    Stores one 1D array of coordinates per spatial direction and provides
    helpers for constructing the full set of grid points or querying grid
    metadata.

    Immutable once constructed: the coordinates are copied into a C++ value on the way in, and
    the arrays this object exposes are read-only views of that value, so what was validated at
    construction stays true for the object's whole life. Coordinates of a dtype other than
    ``float32`` and ``float64`` are stored as ``float64``.

    Attributes:
        _h (object): The C++ lattice (``PointsLatticeF64`` or ``PointsLatticeF32``).
        _pts_per_dir (tuple[npt.NDArray[np.float32 | np.float64], ...]): One
            read-only 1D coordinate array per spatial dimension, views of ``_h``. All arrays share
            the same dtype.
    """

    def __init__(self, pts_per_dir: Iterable[npt.NDArray[np.float32 | np.float64]]) -> None:
        """Initialize the points lattice.

        Args:
            pts_per_dir (Iterable[npt.NDArray[np.float32 | np.float64]]): The points per dimension.
                All points must have the same dtype.

        Raises:
            ValueError: If the dimension is less than 1, the points have different dtypes, an
                array is not 1D, an array is empty, or the coordinates are complex.
        """
        raw = tuple(np.asarray(pts) for pts in pts_per_dir)
        if len(raw) < 1:
            raise ValueError("Points lattice must have at least 1 dimension")
        if not all(pts.dtype == raw[0].dtype for pts in raw):
            raise ValueError("All points must have the same dtype")
        if any(pts.ndim != 1 for pts in raw):
            raise ValueError("All points must be 1D")
        coords = [_as_coordinates(pts) for pts in raw]
        lattice_cls = (
            ext.quad.PointsLatticeF32
            if coords[0].dtype.type is np.float32
            else ext.quad.PointsLatticeF64
        )
        # The class is chosen by dtype above; the list's static type spans both.
        self._set_handle(lattice_cls(cast(Any, coords)))

    def _set_handle(self, handle: PointsLatticeF64 | PointsLatticeF32) -> None:
        """Hold a C++ lattice and the read-only views of its coordinates.

        Args:
            handle (PointsLatticeF64 | PointsLatticeF32): The C++ lattice.
        """
        self._h = handle
        # Views of the C++ value, taken once: the value is immutable, so they never go stale, and
        # reading them is as cheap as reading a Python attribute.
        self._pts_per_dir: tuple[npt.NDArray[np.float32 | np.float64], ...] = tuple(
            handle.pts(d) for d in range(handle.dim)
        )

    @classmethod
    def _from_handle(cls, handle: PointsLatticeF64 | PointsLatticeF32) -> PointsLattice:
        """Wrap a C++ lattice built by the library (no copy, no revalidation).

        Args:
            handle (PointsLatticeF64 | PointsLatticeF32): The C++ lattice.

        Returns:
            PointsLattice: The wrapper.
        """
        lattice = cls.__new__(cls)
        lattice._set_handle(handle)
        return lattice

    def __reduce__(self) -> tuple[type[PointsLattice], tuple[tuple[npt.NDArray[Any], ...]]]:
        """Pickle (and copy) through the constructor, from the coordinates.

        Returns:
            tuple[type[PointsLattice], tuple[tuple[npt.NDArray[Any], ...]]]: The class and its
            constructor argument.
        """
        return (type(self), (self._pts_per_dir,))

    @property
    def dim(self) -> int:
        """Get the dimension of the points lattice.

        Returns:
            int: Number of spatial dimensions.
        """
        return len(self._pts_per_dir)

    @property
    def dtype(self) -> npt.DTypeLike:
        """Get the dtype of the points lattice.

        Returns:
            npt.DTypeLike: The numpy floating-point dtype shared by all
            coordinate arrays.
        """
        return self._pts_per_dir[0].dtype

    @property
    def pts_per_dir(self) -> tuple[npt.NDArray[np.float32 | np.float64], ...]:
        """Get the points per dimension.

        Returns:
            tuple[npt.NDArray[np.float32 | np.float64], ...]: One 1D coordinate
            array for each spatial dimension, read-only.
        """
        # Fresh views rather than the held ones, because ``writeable = False`` stops writes to
        # the data and not changes to the metadata: ``arr.shape = (n, 1)`` reshapes a read-only
        # array in place and would leave the lattice holding a 2-D array while :attr:`dim` still
        # reported 1. A view carries its own shape, so that lands on the caller's copy.
        return tuple(pts.view() for pts in self._pts_per_dir)

    def get_all_points(
        self, order: Literal["C", "F"] = "C"
    ) -> npt.NDArray[np.float32 | np.float64]:
        """Get all points in the points lattice.

        Args:
            order (Literal["C", "F"]): The order of the points. Defaults to "C".
                "C" means the last index varies fastest, "F" means the first index varies fastest.

        Returns:
            npt.NDArray[np.float32 | np.float64]: The dim-dimensional points
                in the lattice. It has shape: (n_pts, dim).
        """
        result: npt.NDArray[np.float32 | np.float64] = self._h.all_points(0 if order == "C" else 1)
        return result


def create_lagrange_points_lattice(
    lagrange_variant: LagrangeVariant,
    n_pts_per_dir: Iterable[int],
    dtype: npt.DTypeLike = np.float64,
) -> PointsLattice:
    """Create a Lagrange points lattice for tensor-product evaluation.

    Builds a :class:`PointsLattice` whose per-direction coordinate arrays are
    the Lagrange nodes of the specified variant on ``[0, 1]``.

    Args:
        lagrange_variant (LagrangeVariant): The variant of the Lagrange basis
            (e.g., equispaced, Gauss-Legendre, Gauss-Lobatto-Legendre, etc.).
        n_pts_per_dir (Iterable[int]): Number of points per spatial dimension.
            Each value must be at least 1.
        dtype (npt.DTypeLike): Floating-point dtype for the coordinates.
            Must be float32 or float64. Defaults to np.float64.

    Returns:
        PointsLattice: A lattice whose per-direction coordinate arrays are the
        Lagrange nodes for the given variant and point counts.

    Raises:
        ValueError: If any value in ``n_pts_per_dir`` is less than 1, ``dtype`` is not float32
            or float64, or ``lagrange_variant`` is not a :class:`~pantr.basis.LagrangeVariant`
            value.
    """
    # Lazy import: pantr.basis imports this module.
    from .basis._basis_tabulate import _variant_code  # noqa: PLC0415

    # operator.index refuses a float count (2.7) rather than truncating it.
    counts = [operator.index(n) for n in n_pts_per_dir]
    build = (
        ext.quad.lagrange_points_lattice_f32
        if _is_float32(dtype)
        else ext.quad.lagrange_points_lattice
    )
    return PointsLattice._from_handle(build(_variant_code(lagrange_variant), counts))


class QuadratureRule:
    """Immutable quadrature rule on the unit cube ``[0, 1]^ndim``.

    Bundles quadrature points and weights as the *reference* rule that
    :func:`pantr.grid.cell_quadrature` affinely maps onto each cell of a grid.
    Points lie in the closed unit cube; the factory-built rules
    (:func:`tensor_product_quadrature`, :func:`gauss_legendre_quadrature`) have
    weights summing to ``1``, the measure of the unit cube, so the rule
    integrates the constant ``1`` to within the rounding of that sum and not
    exactly: a ``ndim``-fold product of rounded 1D weights, summed over
    ``num_points`` terms, cannot land on ``1.0`` bitwise. The stored arrays are read-only views of
    a C++ value.
    """

    __slots__ = ("_h", "_ndim", "_num_points", "_points", "_weights")

    def __init__(self, points: npt.ArrayLike, weights: npt.ArrayLike) -> None:
        """Build and validate a quadrature rule on the unit cube.

        Args:
            points (npt.ArrayLike): ``(num_points, ndim)`` array-like; every
                coordinate must lie in ``[0, 1]``.
            weights (npt.ArrayLike): ``(num_points,)`` array-like of weights.

        Raises:
            ValueError: If ``points`` is not 2D, ``weights`` is not 1D, their
                lengths disagree, either is empty, any value is non-finite, or any
                point lies outside ``[0, 1]``.
        """
        # order="C" rather than ascontiguousarray, which would turn a 0-d array into a 1-d one
        # and hide it from the rank checks.
        pts = np.asarray(points, dtype=np.float64, order="C")
        wts = np.asarray(weights, dtype=np.float64, order="C")
        self._set_handle(ext.quad.QuadratureRule(pts, wts))

    def _set_handle(self, handle: _QuadratureRuleHandle) -> None:
        """Hold a C++ rule and cache what its properties return (it is immutable).

        Args:
            handle (_QuadratureRuleHandle): A C++ ``QuadratureRule``.
        """
        self._h = handle
        self._ndim = int(handle.ndim)
        self._num_points = int(handle.num_points)
        self._points: npt.NDArray[np.float64] = handle.points
        self._weights: npt.NDArray[np.float64] = handle.weights

    @classmethod
    def _from_handle(cls, handle: _QuadratureRuleHandle) -> QuadratureRule:
        """Wrap a C++ rule built by the library (no copy, no revalidation).

        Args:
            handle (_QuadratureRuleHandle): A C++ ``QuadratureRule``.

        Returns:
            QuadratureRule: The wrapper.
        """
        rule = cls.__new__(cls)
        rule._set_handle(handle)
        return rule

    def __reduce__(
        self,
    ) -> tuple[type[QuadratureRule], tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
        """Pickle (and copy) through the constructor, from the points and weights.

        Returns:
            tuple[type[QuadratureRule], tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
            The class and its constructor arguments.
        """
        return (type(self), (self._points, self._weights))

    @property
    def ndim(self) -> int:
        """Get the spatial dimension of the rule.

        Returns:
            int: Number of axes (``>= 1``).
        """
        return self._ndim

    @property
    def num_points(self) -> int:
        """Get the number of quadrature points.

        Returns:
            int: Point count (``>= 1``).
        """
        return self._num_points

    @property
    def points(self) -> npt.NDArray[np.float64]:
        """Get the quadrature points on the unit cube.

        Returns:
            npt.NDArray[np.float64]: Read-only ``(num_points, ndim)`` array in
            ``[0, 1]^ndim``.
        """
        return self._points

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        """Get the quadrature weights.

        Returns:
            npt.NDArray[np.float64]: Read-only ``(num_points,)`` array.
        """
        return self._weights

    def __repr__(self) -> str:
        """Return a compact representation useful for debugging.

        Returns:
            str: ``"QuadratureRule(ndim=..., num_points=...)"``.
        """
        return f"QuadratureRule(ndim={self._ndim}, num_points={self._num_points})"


def tensor_product_quadrature(
    rules: Sequence[tuple[npt.ArrayLike, npt.ArrayLike]],
) -> QuadratureRule:
    """Build a tensor-product :class:`QuadratureRule` from per-axis 1D rules.

    Each axis contributes a 1D rule ``(nodes, weights)`` on ``[0, 1]``; the
    d-dimensional rule is their tensor product. Points are enumerated in
    row-major (C) order -- the last axis varies fastest, matching
    :class:`pantr.grid.TensorProductGrid` cell ids -- and each weight is the
    product of the corresponding per-axis weights.

    Args:
        rules (Sequence[tuple[npt.ArrayLike, npt.ArrayLike]]): One
            ``(nodes, weights)`` pair per axis. Within a pair, ``nodes`` and
            ``weights`` must be 1D of equal, non-zero length, with nodes in
            ``[0, 1]``.

    Returns:
        QuadratureRule: The tensor-product rule on ``[0, 1]^ndim`` with
        ``ndim == len(rules)`` and ``num_points`` the product of the per-axis
        point counts.

    Raises:
        ValueError: If ``rules`` is empty, or any axis pair is not a matching
            pair of non-empty 1D arrays, or (via :class:`QuadratureRule`) any
            node lies outside ``[0, 1]``.
    """
    pairs = [
        (
            np.asarray(pair[0], dtype=np.float64, order="C"),
            np.asarray(pair[1], dtype=np.float64, order="C"),
        )
        for pair in rules
    ]
    return QuadratureRule._from_handle(ext.quad.tensor_product_quadrature(pairs))


def gauss_legendre_quadrature(ndim: int, npts: int | Sequence[int]) -> QuadratureRule:
    """Build a tensor-product Gauss-Legendre :class:`QuadratureRule` on the unit cube.

    Args:
        ndim (int): Number of axes (``>= 1``).
        npts (int | Sequence[int]): Points per axis. A scalar is broadcast to
            every axis; a length-``ndim`` sequence gives per-axis counts. Each
            count must be ``>= 1``.

    Returns:
        QuadratureRule: The tensor product of per-axis ``npts``-point
        Gauss-Legendre rules. Exact for tensor-product polynomials of per-axis
        degree ``2 * npts - 1``; weights sum to ``1``.

    Raises:
        ValueError: If ``ndim < 1``, ``npts`` is a sequence of the wrong length,
            or any count is ``< 1``.

    References:
        Nodes and weights follow the classical Gauss-Legendre construction
        :cite:p:`golub1969gauss`.
    """
    counts = [int(npts)] * ndim if isinstance(npts, int) else [int(n) for n in npts]
    return QuadratureRule._from_handle(ext.quad.gauss_legendre_quadrature(ndim, counts))


__all__ = [
    "PointsLattice",
    "QuadratureRule",
    "create_lagrange_points_lattice",
    "gauss_legendre_quadrature",
    "get_chebyshev_gauss_1st_kind_1d",
    "get_chebyshev_gauss_2nd_kind_1d",
    "get_gauss_legendre_1d",
    "get_gauss_lobatto_legendre_1d",
    "get_modified_chebyshev_nodes_1d",
    "get_tanh_sinh_1d",
    "get_trapezoidal_1d",
    "tensor_product_quadrature",
]
