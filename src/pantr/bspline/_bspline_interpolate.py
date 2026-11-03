"""B-spline interpolation, fitting, and L2 projection.

Provides :func:`interpolate_bspline` (callable-based),
:func:`fit_bspline` (pre-evaluated values), and
:func:`l2_project_bspline` (L2 projection with per-element quadrature).

The systems are built and solved in C++ (``include/pantr/bspline/fit.hpp`` and
``l2_project.hpp``) in their staged form: C++ builds what is to be sampled, this module calls the
user's function on it, and C++ solves for the coefficients. The dtype rules (which type a solve
runs in, which result is refused) are the previous release's NumPy promotions, decided here.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from numpy import typing as npt

from .._ext import ext
from .._interpolation_utils import resolve_svd_tolerance, split_components
from ..quad import PointsLattice
from ._bspline_space_factory import create_greville_lattice
from ._bspline_space_nd import BsplineSpace

if TYPE_CHECKING:
    from ._bspline import Bspline

_cpp: Any = ext.bspline
"""The compiled fitting systems (private; untyped: one class per dtype)."""

_F32: np.dtype[np.float32] = np.dtype(np.float32)
_F64: np.dtype[np.float64] = np.dtype(np.float64)


# ---------------------------------------------------------------------------
# Dtype rules of the previous NumPy implementation
# ---------------------------------------------------------------------------


def _native(dtype: np.dtype[Any]) -> np.dtype[Any]:
    """Return ``dtype`` in native byte order (a byte-swapped float solves as the native one).

    Args:
        dtype (np.dtype): Any dtype.

    Returns:
        np.dtype: The same dtype in native byte order.
    """
    return dtype if dtype.isnative else dtype.newbyteorder("=")


def _is_linalg_float(dtype: np.dtype[Any]) -> bool:
    """Return whether NumPy's ``linalg`` accepted ``dtype`` (float32 or float64).

    Args:
        dtype (np.dtype): A native dtype.

    Returns:
        bool: True for float32 and float64.
    """
    return dtype in (_F32, _F64)


def _unsupported_in_linalg(dtype: np.dtype[Any]) -> TypeError:
    """Return the error ``numpy.linalg.solve`` raised for a float it does not support.

    Args:
        dtype (np.dtype): The unsupported dtype (float16, longdouble).

    Returns:
        TypeError: The error, with NumPy's message.
    """
    return TypeError(f"array type {dtype.name} is unsupported in linalg")


def _linalg_dtype(dtype: np.dtype[Any]) -> np.dtype[Any]:
    """Return the dtype a compiled solve runs in for a NumPy promotion ``dtype``.

    Args:
        dtype (np.dtype): The promotion of a solve's operand dtypes.

    Returns:
        np.dtype: float32 for float32, float64 otherwise.
    """
    return _F32 if dtype == _F32 else _F64


def _tag(dtype: np.dtype[Any]) -> str:
    """Return the suffix naming a float32 or float64 dtype in the compiled names.

    Args:
        dtype (np.dtype): float32 or float64.

    Returns:
        str: ``"f32"`` or ``"f64"``.
    """
    return "f32" if dtype == _F32 else "f64"


def _derivative_count(n: Any) -> int:  # noqa: ANN401
    """Normalize one boundary derivative count as the previous code used it.

    A count that is not positive constrains nothing; a positive one had to be an integer (a
    ``bool`` or a float raised ``TypeError`` from the derivative tabulation).

    Args:
        n (Any): The count as given.

    Returns:
        int: The count, or 0 when it is not positive.

    Raises:
        TypeError: A positive ``bool`` or a positive non-integer.
    """
    if not n > 0:
        return 0
    if isinstance(n, bool):
        raise TypeError("an integer is required")
    return operator.index(n)


def _normalize_boundary_derivatives(
    space: BsplineSpace,
    boundary_derivatives: Sequence[tuple[int, ...] | None] | None,
) -> list[tuple[int, int] | None]:
    """Unpack the per-direction derivative constraints as the previous code did.

    Args:
        space (BsplineSpace): The target space.
        boundary_derivatives: Per-direction ``(n_left, n_right)`` or ``None``.

    Returns:
        list[tuple[int, int] | None]: One entry per direction the sequence reaches; ``None`` for a
        skipped or periodic direction.

    Raises:
        ValueError: An entry does not unpack into two counts.
        TypeError: A count of the wrong type (see :func:`_derivative_count`).
    """
    if boundary_derivatives is None:
        return []
    out: list[tuple[int, int] | None] = []
    for d, s1d in enumerate(space.spaces):
        if d >= len(boundary_derivatives):
            break
        bd = boundary_derivatives[d]
        if bd is None or s1d.periodic:
            out.append(None)
            continue
        n_left, n_right = bd
        out.append((_derivative_count(n_left), _derivative_count(n_right)))
    return out


def _tensor_collocation(
    space: BsplineSpace,
    node_arrays: list[npt.NDArray[np.float32 | np.float64]],
    boundary_derivatives: Sequence[tuple[int, ...] | None] | None,
) -> tuple[Any, np.dtype[Any]]:
    """Build the compiled tensor-product collocation system.

    Args:
        space (BsplineSpace): The target space.
        node_arrays (list[npt.NDArray]): Per-direction nodes.
        boundary_derivatives: Per-direction ``(n_left, n_right)`` or ``None``.

    Returns:
        tuple[Any, np.dtype]: The system and its matrices' dtype (the nodes' dtype).

    Raises:
        ValueError: A node outside the domain, or more derivative rows than nodes allow.
        ZeroDivisionError: The derivative recurrence divided by zero.
    """
    dtype = np.result_type(*node_arrays) if node_arrays else space.dtype
    mat_dtype = _linalg_dtype(np.dtype(dtype))
    arrays = [np.ascontiguousarray(a, dtype=mat_dtype) for a in node_arrays]
    derivs = _normalize_boundary_derivatives(space, boundary_derivatives)
    build = getattr(_cpp, "tensor_collocation_" + _tag(mat_dtype))
    return build(space._h, arrays, derivs), mat_dtype


def _solve_tensor(
    system: Any,  # noqa: ANN401
    mat_dtype: np.dtype[Any],
    values: npt.NDArray[np.floating[Any]],
    value_dtype: np.dtype[Any],
    tol: float | None,
) -> npt.NDArray[np.floating[Any]]:
    """Solve a collocation system for values given on its node grid.

    The solve runs in the promotion of the matrices' and the values' dtypes (float64 where that
    promotion is a float NumPy's ``linalg`` did not support), and the coefficients are returned in
    the values' dtype, as before.

    Args:
        system (Any): The compiled system.
        mat_dtype (np.dtype): Its matrices' dtype.
        values (npt.NDArray): ``(*grid, rank)``.
        value_dtype (np.dtype): The values' (native) dtype.
        tol (float | None): Truncation tolerance; ``100 * eps`` of the matrices' dtype when
            ``None``.

    Returns:
        npt.NDArray: ``(*num_basis, rank)`` coefficients in ``value_dtype``.

    Raises:
        ValueError: No node in some direction (NumPy's reshape message, as before).
    """
    if values.size == 0:
        # The previous solve failed reshaping the empty value tensor; its message named the
        # extent of the direction it was solving, here the first one's.
        raise ValueError(f"cannot reshape array of size 0 into shape ({values.shape[0]},newaxis)")
    compute = _linalg_dtype(np.result_type(mat_dtype, value_dtype))
    rank = int(values.shape[-1])
    resolved = float(resolve_svd_tolerance(mat_dtype, tol))
    solve = system.solve_f32 if compute == _F32 else system.solve_f64
    ctrl = solve(np.ascontiguousarray(values, dtype=compute), rank, resolved)
    return np.asarray(ctrl).astype(value_dtype, copy=False)


# ---------------------------------------------------------------------------
# Function evaluation and node resolution
# ---------------------------------------------------------------------------


def _evaluate_func_on_lattice(
    func: Callable[..., npt.ArrayLike],
    lattice: PointsLattice,
    grid_shape: tuple[int, ...],
) -> tuple[list[npt.NDArray[np.floating[Any]]], np.dtype[np.float32] | np.dtype[np.float64]]:
    """Evaluate a callable on a PointsLattice and split into components.

    Args:
        func (Callable): Function receiving a :class:`PointsLattice`.
        lattice (PointsLattice): Tensor-product sampling grid.
        grid_shape (tuple[int, ...]): Expected grid shape
            ``(n_pts_0, ..., n_pts_{d-1})``.

    Returns:
        tuple: ``(components, dtype)`` where *components* is a list of arrays
        each with shape ``grid_shape``, and *dtype* is the inferred floating
        dtype.

    Raises:
        ValueError: If the callable returns an unexpected shape.
    """
    n_total = int(np.prod(grid_shape))
    raw_untyped = np.asarray(func(lattice))
    if not np.issubdtype(raw_untyped.dtype, np.floating):
        raw_untyped = raw_untyped.astype(np.float64)
    raw: npt.NDArray[np.floating[Any]] = raw_untyped

    if raw.ndim == 1:
        if raw.shape[0] != n_total:
            raise ValueError(
                f"Function returned shape {raw.shape}, expected ({n_total},) or ({n_total}, rank)."
            )
        values = raw.reshape(grid_shape)
    elif raw.ndim == 2:  # noqa: PLR2004
        if raw.shape[0] != n_total:
            raise ValueError(
                f"Function returned shape {raw.shape}, expected ({n_total},) or ({n_total}, rank)."
            )
        values = raw.reshape(*grid_shape, raw.shape[1])
    else:
        raise ValueError(
            f"Function returned shape {raw.shape}, expected ({n_total},) or ({n_total}, rank)."
        )

    components = split_components(values, grid_shape)
    _f32: np.dtype[np.float32] = np.dtype(np.float32)
    _f64: np.dtype[np.float64] = np.dtype(np.float64)
    out_dtype = _f32 if raw.dtype == _f32 else _f64
    return components, out_dtype


def _resolve_nodes(
    space: BsplineSpace,
    nodes: (
        Literal["greville"]
        | PointsLattice
        | npt.NDArray[np.floating[Any]]
        | Sequence[npt.NDArray[np.floating[Any]]]
        | None
    ),
) -> list[npt.NDArray[np.float32 | np.float64]]:
    """Resolve the *nodes* argument into per-direction 1D arrays.

    Args:
        space (BsplineSpace): The target B-spline space.
        nodes: Node specification (see :func:`interpolate_bspline`).

    Returns:
        list[npt.NDArray]: One 1D node array per parametric direction.

    Raises:
        ValueError: If nodes are inconsistent with the space.
    """
    if nodes is None or (isinstance(nodes, str) and nodes == "greville"):
        lattice = create_greville_lattice(space)
        return list(lattice.pts_per_dir)

    if isinstance(nodes, PointsLattice):
        if nodes.dim != space.dim:
            raise ValueError(f"PointsLattice has {nodes.dim} dimensions, expected {space.dim}.")
        return list(nodes.pts_per_dir)

    if isinstance(nodes, np.ndarray) and nodes.ndim == 1:
        if space.dim != 1:
            raise ValueError(f"A single 1D array implies 1D, but space has {space.dim} dimensions.")
        return [nodes.astype(space.dtype, copy=False)]

    # Sequence of 1D arrays.
    node_list = list(nodes)
    if len(node_list) != space.dim:
        raise ValueError(f"Expected {space.dim} node arrays, got {len(node_list)}.")
    return [np.asarray(a, dtype=space.dtype) for a in node_list]


def _is_scattered_nodes(
    nodes: (
        PointsLattice | npt.NDArray[np.floating[Any]] | Sequence[npt.NDArray[np.floating[Any]]]
    ),
) -> bool:
    """Check whether *nodes* represents scattered (non-tensor-product) points.

    Args:
        nodes: The nodes argument.

    Returns:
        bool: ``True`` if *nodes* is a 2D ndarray (scattered), ``False``
        otherwise.
    """
    if isinstance(nodes, PointsLattice):
        return False
    return isinstance(nodes, np.ndarray) and nodes.ndim == 2  # noqa: PLR2004


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------


def interpolate_bspline(
    func: Callable[..., npt.ArrayLike],
    space: BsplineSpace,
    *,
    nodes: (
        Literal["greville"]
        | PointsLattice
        | npt.NDArray[np.floating[Any]]
        | Sequence[npt.NDArray[np.floating[Any]]]
        | None
    ) = None,
    boundary_derivatives: Sequence[tuple[int, ...] | None] | None = None,
    tol: float | None = None,
) -> Bspline:
    """Interpolate a callable onto a B-spline space.

    Evaluate ``func`` on a tensor-product grid of interpolation nodes and
    recover B-spline coefficients by solving per-direction collocation systems
    (Kronecker structure).

    Args:
        func (Callable[..., npt.ArrayLike]): Function to interpolate.
            Called as ``func(lattice)`` where ``lattice`` is a
            :class:`~pantr.quad.PointsLattice` representing the
            tensor-product sampling grid. Must return an array of shape
            ``(n_total,)`` for scalar or ``(n_total, rank)`` for
            vector-valued functions, where ``n_total = prod(n_pts)``.
        space (~pantr.bspline.BsplineSpace): The target B-spline space.
        nodes: Interpolation node selection.

            - ``None`` or ``"greville"`` (default): Greville abscissae
              (one per basis function per direction).
            - A :class:`~pantr.quad.PointsLattice`: custom
              tensor-product grid.
            - A 1D ``ndarray``: custom nodes for a 1D space.
            - A sequence of 1D ``ndarray`` values: per-direction custom
              nodes.
        boundary_derivatives (Sequence[tuple[int, ...] | None] | None):
            Per-direction boundary derivative constraints. Each entry is
            ``(n_left, n_right)`` specifying how many derivative orders
            to constrain at left/right boundaries. The corresponding
            derivatives are set to zero. ``None`` entries skip that
            direction. Ignored for periodic directions. Defaults to
            ``None`` (no derivative constraints).
        tol (float | None): SVD truncation tolerance for the
            collocation solve. Singular values below
            ``tol * sigma_max`` are treated as zero. If *None*,
            defaults to ``100 * machine_epsilon``. Only affects
            overdetermined or near-singular systems; square
            well-conditioned systems use a direct solve.

    Returns:
        Bspline: A non-rational B-spline whose evaluation approximates
        ``func``.

    Raises:
        TypeError: If ``space`` is not a :class:`BsplineSpace`.
        ValueError: If ``nodes`` is inconsistent with ``space``, or the
            callable returns an unexpected shape.

    Example:
        >>> import numpy as np
        >>> from pantr.bspline import create_uniform_space, interpolate_bspline
        >>> space = create_uniform_space(3, 4)
        >>> b = interpolate_bspline(
        ...     lambda lat: lat.pts_per_dir[0] ** 2, space
        ... )
        >>> b.degree
        (3,)
    """
    from ._bspline import Bspline  # noqa: PLC0415

    if not isinstance(space, BsplineSpace):
        raise TypeError(f"Expected BsplineSpace, got {type(space).__name__}")

    node_arrays = _resolve_nodes(space, nodes)

    # The collocation matrices (with derivative rows) are built before the function is sampled.
    system, mat_dtype = _tensor_collocation(space, node_arrays, boundary_derivatives)

    lattice = PointsLattice(node_arrays)
    grid_shape = tuple(a.shape[0] for a in node_arrays)
    components, out_dtype = _evaluate_func_on_lattice(func, lattice, grid_shape)

    values = np.stack(components, axis=-1)
    ctrl = _solve_tensor(system, mat_dtype, values, out_dtype, tol)
    return Bspline(space, ctrl)


# ---------------------------------------------------------------------------
# Fit (from pre-evaluated values)
# ---------------------------------------------------------------------------


def fit_bspline(
    values: npt.ArrayLike,
    nodes: (
        PointsLattice | npt.NDArray[np.floating[Any]] | Sequence[npt.NDArray[np.floating[Any]]]
    ),
    space: BsplineSpace,
    *,
    tol: float | None = None,
) -> Bspline:
    """Construct a B-spline from pre-evaluated sample values at known nodes.

    Recover B-spline coefficients by solving the collocation system.
    For tensor-product nodes, uses per-direction solves (Kronecker
    structure). For scattered nodes, builds the full collocation matrix
    and solves it as a single system.

    The output dtype is inferred from *values*.

    Supports two point layouts:

    - **Tensor-product** (a :class:`~pantr.quad.PointsLattice`, a single
      1D array, or a sequence of 1D arrays): ``values`` must have shape
      ``(*n_pts_per_dir)`` (scalar) or ``(*n_pts_per_dir, rank)``
      (vector).
    - **Scattered** (a 2D ``ndarray`` of shape ``(n_pts, dim)``):
      ``values`` must have shape ``(n_pts,)`` (scalar) or
      ``(n_pts, rank)`` (vector).

    Args:
        values (npt.ArrayLike): Sample values at the nodes.
        nodes: Interpolation nodes.

            - A :class:`~pantr.quad.PointsLattice`: tensor-product grid.
            - A 1D ``ndarray``: 1D tensor-product (single direction).
            - A sequence of 1D ``ndarray`` values: N-D tensor-product.
            - A 2D ``ndarray`` of shape ``(n_pts, dim)``: scattered
              points.
        space (~pantr.bspline.BsplineSpace): The target B-spline space.
        tol (float | None): SVD truncation tolerance. If *None*,
            defaults to ``100 * machine_epsilon``. Only affects
            overdetermined or near-singular systems; square
            well-conditioned systems use a direct solve.

    Returns:
        Bspline: A non-rational B-spline.

    Raises:
        TypeError: If ``space`` is not a :class:`BsplineSpace`.
        ValueError: If *nodes* are inconsistent with *space*, or the
            system is underdetermined for scattered nodes.

    Example:
        >>> import numpy as np
        >>> from pantr.bspline import create_uniform_space, fit_bspline
        >>> space = create_uniform_space(3, 4)
        >>> nodes = np.linspace(0, 1, 20)
        >>> vals = np.sin(nodes)
        >>> b = fit_bspline(vals, [nodes], space)
    """
    if not isinstance(space, BsplineSpace):
        raise TypeError(f"Expected BsplineSpace, got {type(space).__name__}")

    vals = np.asarray(values)
    if not np.issubdtype(vals.dtype, np.floating):
        vals = vals.astype(np.float64)

    if _is_scattered_nodes(nodes):
        return _fit_from_scattered(vals, nodes, space, tol)  # type: ignore[arg-type]

    return _fit_from_tensor_product(vals, nodes, space, tol)


def _fit_from_tensor_product(
    values: npt.NDArray[np.floating[Any]],
    nodes: (
        PointsLattice | npt.NDArray[np.floating[Any]] | Sequence[npt.NDArray[np.floating[Any]]]
    ),
    space: BsplineSpace,
    tol: float | None,
) -> Bspline:
    """Fit from tensor-product nodes via Kronecker solve.

    Args:
        values (npt.NDArray): Sample values.
        nodes: Tensor-product nodes.
        space (BsplineSpace): Target B-spline space.
        tol (float | None): SVD truncation tolerance.

    Returns:
        Bspline: The fitted B-spline.

    Raises:
        TypeError: Values of a float type NumPy's ``linalg`` does not support, when the first
            direction's system is square.
        ValueError: Inconsistent nodes or values, a node outside the domain, or values of
            another dtype than the space.
    """
    from ._bspline import Bspline  # noqa: PLC0415

    # Resolve nodes to per-direction arrays.
    if isinstance(nodes, PointsLattice):
        if nodes.dim != space.dim:
            raise ValueError(f"PointsLattice has {nodes.dim} dimensions, expected {space.dim}.")
        node_arrays = list(nodes.pts_per_dir)
    elif isinstance(nodes, np.ndarray) and nodes.ndim == 1:
        if space.dim != 1:
            raise ValueError(f"A single 1D array implies 1D, but space has {space.dim} dimensions.")
        node_arrays = [nodes.astype(space.dtype, copy=False)]
    else:
        node_arrays = [np.asarray(a, dtype=space.dtype) for a in nodes]
        if len(node_arrays) != space.dim:
            raise ValueError(f"Expected {space.dim} node arrays, got {len(node_arrays)}.")

    grid_shape = tuple(a.shape[0] for a in node_arrays)
    components = split_components(values, grid_shape)
    system, mat_dtype = _tensor_collocation(space, node_arrays, None)

    value_dtype = _native(values.dtype)
    # A float NumPy's linalg does not support: the first direction's square system went to
    # numpy.linalg.solve, which refused it; otherwise the coefficients came back in that dtype and
    # the constructor refuses them.
    if not _is_linalg_float(value_dtype) and grid_shape[0] == space.spaces[0].num_basis:
        raise _unsupported_in_linalg(value_dtype)

    stacked = np.stack(components, axis=-1)
    ctrl = _solve_tensor(system, mat_dtype, stacked, value_dtype, tol)
    return Bspline(space, ctrl)


def _fit_from_scattered(
    values: npt.NDArray[np.floating[Any]],
    nodes: npt.NDArray[np.floating[Any]],
    space: BsplineSpace,
    tol: float | None,
) -> Bspline:
    """Fit from scattered (non-tensor-product) nodes via the full collocation solve.

    Args:
        values (npt.NDArray): Sample values of shape ``(n_pts,)`` or
            ``(n_pts, rank)``.
        nodes (npt.NDArray): Scattered nodes of shape ``(n_pts, dim)``.
        space (BsplineSpace): Target B-spline space.
        tol (float | None): SVD truncation tolerance.

    Returns:
        Bspline: The fitted B-spline.

    Raises:
        TypeError: Nodes, or values of a square system, of a float type NumPy's ``linalg`` does
            not support.
        ValueError: If the system is underdetermined, the nodes have another number of columns
            or are outside the domain, integer nodes come without a ``tol``, or the values do not
            match the nodes.
    """
    from ._bspline import Bspline  # noqa: PLC0415

    n_pts = nodes.shape[0]
    ndim = nodes.shape[1]
    if ndim != space.dim:
        raise ValueError(f"Scattered nodes have {ndim} columns, expected {space.dim}.")
    n_total_basis = space.num_total_basis
    if n_pts < n_total_basis:
        raise ValueError(
            f"Underdetermined system: {n_pts} points < {n_total_basis} basis functions."
        )
    if values.ndim not in (1, 2):
        raise ValueError(
            f"Values have shape {values.shape}, expected ({n_pts},) (scalar) or "
            f"({n_pts}, 'rank') (vector)."
        )

    # The previous code built the matrix in the nodes' dtype: integer nodes reached the default
    # tolerance's machine epsilon, which an integer type has none of, and a float type NumPy's
    # linalg does not support reached the solve.
    node_dtype = _native(nodes.dtype)
    if not np.issubdtype(node_dtype, np.inexact):
        if tol is None:
            raise ValueError(f"data type {node_dtype.type!r} not inexact")
        node_dtype = _F64
    elif not _is_linalg_float(node_dtype):
        raise _unsupported_in_linalg(node_dtype)

    value_dtype = _native(values.dtype)
    if not _is_linalg_float(value_dtype) and n_pts == n_total_basis:
        raise _unsupported_in_linalg(value_dtype)  # the square system went to linalg.solve
    rank = 1 if values.ndim == 1 else int(values.shape[1])
    # The coefficients come back in the promotion (float16 values promote to the nodes' dtype;
    # a longdouble promotion is refused by the constructor, as before).
    result_dtype = np.result_type(node_dtype, value_dtype)
    compute = _linalg_dtype(result_dtype)
    resolved = float(resolve_svd_tolerance(node_dtype, tol))
    pts = np.ascontiguousarray(nodes, dtype=node_dtype)
    vals = np.ascontiguousarray(values.reshape(n_pts, rank), dtype=compute)
    fit = getattr(_cpp, f"fit_scattered_{_tag(node_dtype)}_{_tag(compute)}")
    ctrl = fit(space._h, pts, vals, resolved)
    return Bspline(space, np.asarray(ctrl).astype(result_dtype, copy=False))


# ---------------------------------------------------------------------------
# L2 projection
# ---------------------------------------------------------------------------


def l2_project_bspline(  # noqa: PLR0913
    func: Callable[..., npt.ArrayLike],
    space: BsplineSpace,
    *,
    n_quad: int | Sequence[int] | None = None,
    quadrature: Literal["gauss-legendre", "gauss-lobatto"] = "gauss-legendre",
    boundary_interpolation: bool | Sequence[tuple[bool, bool]] = False,
    tol: float | None = None,
) -> Bspline:
    """L2-project a callable function onto a B-spline space.

    Assemble per-direction mass matrices and load vectors using per-element
    quadrature, then solve the normal equations via sequential 1D solves
    (Kronecker structure).

    The output dtype is inferred from the return value of ``func``.

    Args:
        func (Callable[..., npt.ArrayLike]): Function to project.
            Called as ``func(lattice)`` where ``lattice`` is a
            :class:`~pantr.quad.PointsLattice` of quadrature points.
            Must return an array of shape ``(n_total,)`` for scalar or
            ``(n_total, rank)`` for vector-valued functions.
        space (~pantr.bspline.BsplineSpace): The target B-spline space.
        n_quad (int | Sequence[int] | None): Quadrature points per
            element per direction. Defaults to ``degree + 1``.
        quadrature (Literal["gauss-legendre", "gauss-lobatto"]):
            Quadrature rule type. Defaults to ``"gauss-legendre"``.
        boundary_interpolation (bool | Sequence[tuple[bool, bool]]):
            Replace boundary rows with interpolation conditions.

            - ``False`` (default): pure L2 projection.
            - ``True``: interpolate at all non-periodic boundaries.
            - A sequence of ``(left, right)`` bool pairs: per-direction
              boundary flags.
        tol (float | None): SVD truncation tolerance. If *None*,
            defaults to ``100 * machine_epsilon``.

    Returns:
        Bspline: A non-rational B-spline whose evaluation is the L2
        best approximation of ``func`` in the given space.

    Raises:
        TypeError: If ``space`` is not a :class:`BsplineSpace`.
        ValueError: If ``n_quad`` or ``boundary_interpolation`` is
            inconsistent with ``space``.

    Example:
        >>> import numpy as np
        >>> from pantr.bspline import create_uniform_space, l2_project_bspline
        >>> space = create_uniform_space(3, 4)
        >>> b = l2_project_bspline(
        ...     lambda lat: lat.pts_per_dir[0] ** 2, space
        ... )
    """
    from ._bspline import Bspline  # noqa: PLC0415

    if not isinstance(space, BsplineSpace):
        raise TypeError(f"Expected BsplineSpace, got {type(space).__name__}")

    projection, _, _ = _l2_projection(space, n_quad, quadrature, boundary_interpolation)

    # Sample the function on the quadrature lattice.
    quad_nodes_per_dir = list(projection.nodes)
    quad_lattice = PointsLattice(quad_nodes_per_dir)
    quad_grid_shape = tuple(a.shape[0] for a in quad_nodes_per_dir)
    components, out_dtype = _evaluate_func_on_lattice(func, quad_lattice, quad_grid_shape)

    compute = np.result_type(space.dtype, out_dtype)
    resolved = float(resolve_svd_tolerance(space.dtype, tol))
    n_traces = len(projection.traces)
    n_components = len(components)
    ctrl_components: list[npt.NDArray[np.floating[Any]]] = []
    for comp_idx, comp in enumerate(components):
        # The traces are sampled per component, in the order the previous code sampled them.
        traces = [
            _trace_values(func, projection, t, comp_idx, n_components, compute)
            for t in range(n_traces)
        ]
        project = projection.project_f32 if compute == _F32 else projection.project_f64
        ctrl_components.append(project(np.ascontiguousarray(comp, dtype=compute), traces, resolved))

    ctrl = np.stack(ctrl_components, axis=-1)
    return Bspline(space, ctrl)


def _l2_projection(
    space: BsplineSpace,
    n_quad: int | Sequence[int] | None,
    quadrature: Literal["gauss-legendre", "gauss-lobatto"],
    boundary_interpolation: bool | Sequence[tuple[bool, bool]],
) -> tuple[Any, list[tuple[bool, bool]], tuple[int, ...]]:
    """Resolve the L2 settings as before and build the compiled projection.

    Args:
        space (BsplineSpace): The target space.
        n_quad (int | Sequence[int] | None): Quadrature points per element per direction.
        quadrature (Literal["gauss-legendre", "gauss-lobatto"]): Any value other than
            ``"gauss-legendre"`` selects Gauss-Lobatto, as before.
        boundary_interpolation (bool | Sequence[tuple[bool, bool]]): Boundary flags.

    Returns:
        tuple: ``(projection, bi_flags, n_quads)``: the compiled projection, the resolved
        per-direction flags and the resolved point counts.

    Raises:
        ValueError: If ``n_quad`` or ``boundary_interpolation`` is inconsistent with ``space``,
            or a count is below the rule's minimum.
        TypeError: ``n_quad`` or a flag entry of the wrong type.
    """
    ndim = space.dim
    if n_quad is None:
        n_quads = tuple(s.degree + 1 for s in space.spaces)
    elif isinstance(n_quad, int):
        n_quads = tuple(n_quad for _ in range(ndim))
    else:
        n_quads = tuple(n_quad)
        if len(n_quads) != ndim:
            raise ValueError(f"n_quad has length {len(n_quads)}, expected {ndim}")

    bi_flags = _resolve_boundary_interpolation(boundary_interpolation, space)
    flags: list[tuple[bool, bool]] = []
    for d in range(ndim):
        bi_left, bi_right = bi_flags[d]
        flags.append((bool(bi_left), bool(bi_right)))

    cls = _cpp.L2ProjectionF32 if space.dtype == _F32 else _cpp.L2ProjectionF64
    projection = cls(space._h, list(n_quads), quadrature != "gauss-legendre", flags)
    return projection, bi_flags, n_quads


def _trace_values(  # noqa: PLR0913
    func: Callable[..., npt.ArrayLike],
    projection: Any,  # noqa: ANN401
    trace: int,
    comp_idx: int,
    n_components: int,
    dtype: np.dtype[Any],
) -> npt.NDArray[np.floating[Any]]:
    """Sample one component of ``func`` on a boundary trace's lattice.

    Args:
        func (Callable): Function being projected.
        projection (Any): The compiled projection.
        trace (int): The trace's index.
        comp_idx (int): Which output component to extract.
        n_components (int): Total number of output components.
        dtype (np.dtype): The computation dtype.

    Returns:
        npt.NDArray: The component's values over the trace lattice, contiguous in ``dtype``.

    Raises:
        ValueError: The function returned an array that does not reshape to the lattice.
    """
    boundary_nodes = list(projection.trace_nodes(trace))
    lattice = PointsLattice(boundary_nodes)
    grid_shape = tuple(a.shape[0] for a in boundary_nodes)

    raw = np.asarray(func(lattice))
    if not np.issubdtype(raw.dtype, np.floating):
        raw = raw.astype(np.float64)

    if n_components > 1:
        values = raw.reshape(*grid_shape, n_components)[..., comp_idx]
    else:
        values = raw.reshape(grid_shape)
    return np.ascontiguousarray(values, dtype=dtype)


# ---------------------------------------------------------------------------
# The boundary interpolation flags
# ---------------------------------------------------------------------------


def _resolve_boundary_interpolation(
    boundary_interpolation: bool | Sequence[tuple[bool, bool]],
    space: BsplineSpace,
) -> list[tuple[bool, bool]]:
    """Normalize boundary interpolation flags to per-direction pairs.

    Args:
        boundary_interpolation: Flags as provided by the user.
        space (BsplineSpace): The target B-spline space.

    Returns:
        list[tuple[bool, bool]]: Per-direction ``(left, right)`` flags.
    """
    ndim = space.dim
    if isinstance(boundary_interpolation, bool):
        if boundary_interpolation:
            return [(not s.periodic, not s.periodic) for s in space.spaces]
        return [(False, False)] * ndim

    flags = list(boundary_interpolation)
    if len(flags) != ndim:
        raise ValueError(f"boundary_interpolation has length {len(flags)}, expected {ndim}")
    return flags
