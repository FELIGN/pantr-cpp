"""Bernstein interpolation and fitting: construct a Bézier from function samples or values.

Provides :func:`interpolate_bezier` (callable-based) and :func:`fit_bezier`
(pre-evaluated values), which recover the Bernstein coefficients from the
Bernstein Vandermonde system.

The Bernstein Vandermonde matrix becomes increasingly ill-conditioned as the
polynomial degree grows, making a direct solve numerically unstable.  To handle
this, both functions use a **truncated SVD** pseudo-inverse: singular values
smaller than ``tol * sigma_max`` are zeroed out, which regularises the inversion
and yields robust results even at high degree.  The solve runs in pantr's C++
library (``include/pantr/bezier/fit.hpp``); this module resolves the arguments.

Both functions support **tensor-product** grids (represented as a
:class:`~pantr.quad.PointsLattice` or a sequence of 1D node arrays) and
**scattered** point sets (represented as a 2D array of shape
``(n_pts, dim)``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt

from .._ext import ext
from ..quad import PointsLattice, get_modified_chebyshev_nodes_1d

if TYPE_CHECKING:
    from . import Bezier

_cpp: Any = ext.bezier
"""The compiled Bezier fitting (private; untyped: one function per dtype)."""


def _compute_dtype(dtype: np.dtype[Any]) -> np.dtype[np.float32] | np.dtype[np.float64]:
    """Get the dtype a fit runs in: ``float32`` stays, every other real dtype is ``float64``.

    Args:
        dtype (np.dtype[Any]): The values' (floating) dtype.

    Returns:
        np.dtype[np.float32] | np.dtype[np.float64]: The compute dtype.
    """
    if dtype == np.float32:
        return np.dtype(np.float32)
    return np.dtype(np.float64)


def _fit_tensor(
    values: npt.NDArray[np.floating[Any]],
    node_arrays: list[npt.NDArray[np.floating[Any]]],
    degree_tuple: tuple[int, ...] | None,
    tol: float | None,
    dtype: np.dtype[np.float32] | np.dtype[np.float64],
) -> Bezier:
    """Fit Bernstein coefficients on a tensor-product grid (one pseudo-inverse per direction).

    Args:
        values (npt.NDArray[np.floating[Any]]): Shape ``n_pts`` (scalar) or ``(*n_pts, rank)``,
            ``n_pts`` the node counts.
        node_arrays (list[npt.NDArray[np.floating[Any]]]): One 1D node array per direction.
        degree_tuple (tuple[int, ...] | None): Degree per direction, or *None* for
            interpolation (``degree = n_pts - 1``).
        tol (float | None): SVD truncation tolerance.
        dtype (np.dtype): The dtype the fit runs in.

    Returns:
        ~pantr.bezier.Bezier: A non-rational Bézier.
    """
    from . import Bezier as BezierCls  # noqa: PLC0415

    stacked = values if values.ndim == len(node_arrays) + 1 else values[..., np.newaxis]
    vals = np.ascontiguousarray(stacked, dtype=dtype)
    nodes = [np.ascontiguousarray(a, dtype=dtype) for a in node_arrays]
    fit = _cpp.fit_tensor_f32 if dtype == np.float32 else _cpp.fit_tensor
    degrees = None if degree_tuple is None else list(degree_tuple)
    return BezierCls._from_handle(fit(vals, nodes, degrees, tol))


def _resolve_nodes(
    n_pts: tuple[int, ...],
    nodes: (
        Literal["chebyshev", "uniform"]
        | PointsLattice
        | npt.NDArray[np.floating[Any]]
        | Sequence[npt.NDArray[np.floating[Any]]]
        | None
    ),
    dtype: np.dtype[np.float32] | np.dtype[np.float64],
) -> list[npt.NDArray[np.floating[Any]]]:
    """Resolve the *nodes* parameter into a list of 1D node arrays.

    Args:
        n_pts (tuple[int, ...]): Number of sample points per direction.
        nodes: Node specification — ``None`` or ``"chebyshev"`` for modified
            Chebyshev-Lobatto nodes, ``"uniform"`` for equispaced nodes, a
            :class:`~pantr.quad.PointsLattice`, a single 1D array (broadcast
            to every direction), or a sequence of 1D arrays (one per
            direction).
        dtype (np.dtype): Floating dtype for generated nodes.

    Returns:
        list[npt.NDArray[np.floating[Any]]]: One 1D node array per parametric
        direction.

    Raises:
        ValueError: If *nodes* is inconsistent with *n_pts*.
    """
    if nodes is None or (isinstance(nodes, str) and nodes == "chebyshev"):
        return [
            get_modified_chebyshev_nodes_1d(n, dtype)
            if n >= 2  # noqa: PLR2004
            else np.array([0.5], dtype=dtype)
            for n in n_pts
        ]

    if isinstance(nodes, str) and nodes == "uniform":
        result: list[npt.NDArray[np.floating[Any]]] = [
            np.linspace(0.0, 1.0, n, dtype=dtype) for n in n_pts
        ]
        return result

    # PointsLattice
    if isinstance(nodes, PointsLattice):
        if nodes.dim != len(n_pts):
            raise ValueError(f"PointsLattice has {nodes.dim} dimensions, expected {len(n_pts)}.")
        node_list: list[npt.NDArray[np.floating[Any]]] = [
            np.asarray(a, dtype=dtype) for a in nodes.pts_per_dir
        ]
        for i, (arr_i, n) in enumerate(zip(node_list, n_pts, strict=True)):
            if arr_i.shape[0] != n:
                raise ValueError(
                    f"PointsLattice direction {i} has {arr_i.shape[0]} nodes, expected n_pts={n}."
                )
        return node_list

    # User-provided nodes
    if isinstance(nodes, np.ndarray) and nodes.ndim == 1:
        # Single array — broadcast to all directions
        arr: npt.NDArray[np.floating[Any]] = nodes.astype(dtype, copy=False)
        for n in n_pts:
            if arr.shape[0] != n:
                raise ValueError(f"Node array length {arr.shape[0]} does not match n_pts={n}.")
        return [arr] * len(n_pts)

    # Sequence of arrays
    node_list_seq: list[npt.NDArray[np.floating[Any]]] = [np.asarray(a, dtype=dtype) for a in nodes]
    if len(node_list_seq) != len(n_pts):
        raise ValueError(f"Expected {len(n_pts)} node arrays, got {len(node_list_seq)}.")
    for i, (arr, n) in enumerate(zip(node_list_seq, n_pts, strict=True)):
        if arr.ndim != 1 or arr.shape[0] != n:
            raise ValueError(
                f"Node array for direction {i} has shape {arr.shape}, expected ({n},)."
            )
    return node_list_seq


def _validate_n_pts(
    n_pts: int | Sequence[int],
) -> tuple[int, ...]:
    """Normalise and validate *n_pts* to a tuple of positive integers.

    Args:
        n_pts (int | Sequence[int]): Number of sample points per direction.

    Returns:
        tuple[int, ...]: Validated tuple.

    Raises:
        ValueError: If any value is < 1.
    """
    n_pts_tuple = (n_pts,) if isinstance(n_pts, int) else tuple(n_pts)
    for i, n in enumerate(n_pts_tuple):
        if n < 1:
            raise ValueError(f"n_pts[{i}] must be >= 1, got {n}.")
    return n_pts_tuple


def _validate_degree(
    degree: int | Sequence[int] | None,
    n_pts_tuple: tuple[int, ...],
) -> tuple[int, ...] | None:
    """Normalise and validate the *degree* parameter.

    Args:
        degree (int | Sequence[int] | None): Requested degree per direction.
        n_pts_tuple (tuple[int, ...]): Number of sample points per direction.

    Returns:
        tuple[int, ...] | None: Validated degree tuple, or *None* if degree
        was not specified (meaning degree = n_pts - 1).

    Raises:
        ValueError: If degree >= n_pts in any direction.
    """
    if degree is None:
        return None
    deg_tuple = (degree,) * len(n_pts_tuple) if isinstance(degree, int) else tuple(degree)
    if len(deg_tuple) != len(n_pts_tuple):
        raise ValueError(f"degree has {len(deg_tuple)} entries but n_pts has {len(n_pts_tuple)}.")
    for i, (d, n) in enumerate(zip(deg_tuple, n_pts_tuple, strict=True)):
        if d < 0:
            raise ValueError(f"degree[{i}] must be >= 0, got {d}.")
        if d >= n:
            raise ValueError(
                f"degree[{i}]={d} must be < n_pts[{i}]={n} "
                f"(need at least degree + 1 sample points)."
            )
    return deg_tuple


def interpolate_bezier(
    func: Callable[..., npt.ArrayLike],
    n_pts: int | Sequence[int],
    *,
    degree: int | Sequence[int] | None = None,
    nodes: (
        Literal["chebyshev", "uniform"]
        | PointsLattice
        | npt.NDArray[np.floating[Any]]
        | Sequence[npt.NDArray[np.floating[Any]]]
        | None
    ) = None,
    tol: float | None = None,
) -> Bezier:
    """Interpolate a callable function into a Bézier in Bernstein form.

    Evaluates ``func`` on a tensor-product grid of interpolation nodes and
    recovers the Bernstein coefficients via truncated SVD.

    The parametric dimension is determined by the length of ``n_pts`` (when
    given as a sequence) or defaults to 1 (when given as a scalar ``int``).

    The output dtype is inferred from the return value of ``func``.

    Args:
        func (Callable[..., npt.ArrayLike]): Function to interpolate.  Called
            as ``func(lattice)`` where ``lattice`` is a
            :class:`~pantr.quad.PointsLattice` representing the tensor-product
            sampling grid.  The callable may also accept a plain ``ndarray``
            (for compatibility with :meth:`Bezier.evaluate` signatures).
            Must return an array of shape ``(n_total,)`` for a scalar-valued
            function or ``(n_total, rank)`` for a vector-valued function,
            where ``n_total = prod(n_pts)``.
        n_pts (int | Sequence[int]): Number of sample points per parametric
            direction.  A single ``int`` gives a 1D Bézier.
        degree (int | Sequence[int] | None): Polynomial degree per direction.
            If *None* (default), ``degree = n_pts - 1`` (exact interpolation).
            If provided, must satisfy ``degree < n_pts`` in each direction;
            the result is a least-squares approximation.
        nodes: Interpolation node selection.

            - ``None`` or ``"chebyshev"`` (default): modified Chebyshev-Lobatto
              nodes on [0, 1].
            - ``"uniform"``: equispaced nodes on [0, 1].
            - A :class:`~pantr.quad.PointsLattice`: custom tensor-product grid.
            - A 1D ``ndarray``: custom nodes broadcast to all directions.
            - A sequence of 1D ``ndarray`` values: per-direction custom nodes.
        tol (float | None): SVD truncation tolerance. If *None*, uses a
            default based on machine epsilon.

    Returns:
        ~pantr.bezier.Bezier: A non-rational Bézier whose evaluation
        approximates ``func``.

    Raises:
        ValueError: If ``n_pts`` values are < 1, *degree* >= *n_pts*, *nodes*
            is inconsistent with *n_pts*, or the callable returns an
            unexpected shape; or ("SVD did not converge") if a custom node is
            NaN or infinite in a direction with more than one node.

    Example:
        >>> from pantr.bezier import Bezier
        >>> import numpy as np
        >>> b = interpolate_bezier(lambda lattice: lattice.get_all_points()[:, 0] ** 2, [5])
        >>> b.degree
        (4,)
    """
    n_pts_tuple = _validate_n_pts(n_pts)
    degree_tuple = _validate_degree(degree, n_pts_tuple)

    # Resolve nodes and build PointsLattice (default to float64 for generated nodes)
    node_arrays = _resolve_nodes(n_pts_tuple, nodes, np.dtype(np.float64))
    lattice = PointsLattice(node_arrays)
    n_total = int(np.prod(n_pts_tuple))

    # Evaluate function and infer dtype from the return value
    raw_untyped = np.asarray(func(lattice))
    if not np.issubdtype(raw_untyped.dtype, np.floating):
        raw_untyped = raw_untyped.astype(np.float64)
    raw: npt.NDArray[np.floating[Any]] = raw_untyped

    if raw.ndim == 1:
        if raw.shape[0] != n_total:
            raise ValueError(
                f"Function returned shape {raw.shape}, expected ({n_total},) or ({n_total}, rank)."
            )
        values = raw.reshape(n_pts_tuple)
    elif raw.ndim == 2:  # noqa: PLR2004
        if raw.shape[0] != n_total:
            raise ValueError(
                f"Function returned shape {raw.shape}, expected ({n_total},) or ({n_total}, rank)."
            )
        values = raw.reshape(*n_pts_tuple, raw.shape[1])
    else:
        raise ValueError(
            f"Function returned shape {raw.shape}, expected ({n_total},) or ({n_total}, rank)."
        )

    # The nodes are float64, and a pseudo-inverse built on them promotes the values: the fit
    # runs in float64 unless no direction needs one (every direction a single point).
    dtype = _compute_dtype(raw.dtype) if all(n == 1 for n in n_pts_tuple) else np.dtype(np.float64)
    return _fit_tensor(values, node_arrays, degree_tuple, tol, dtype)


def _resolve_nodes_from_user(
    nodes: PointsLattice | npt.NDArray[np.floating[Any]] | Sequence[npt.NDArray[np.floating[Any]]],
    n_pts_tuple: tuple[int, ...],
    dtype: np.dtype[np.float32] | np.dtype[np.float64],
) -> list[npt.NDArray[np.floating[Any]]]:
    """Resolve user-provided tensor-product nodes for :func:`fit_bezier`.

    Args:
        nodes: A :class:`~pantr.quad.PointsLattice`, a single 1D array
            (for 1D fitting), or a sequence of 1D arrays.
        n_pts_tuple (tuple[int, ...]): Expected number of points per direction.
        dtype (np.dtype): Floating dtype.

    Returns:
        list[npt.NDArray[np.floating[Any]]]: One 1D node array per direction.

    Raises:
        ValueError: If nodes are inconsistent with the expected grid shape.
    """
    if isinstance(nodes, PointsLattice):
        if nodes.dim != len(n_pts_tuple):
            raise ValueError(
                f"PointsLattice has {nodes.dim} dimensions, but values have "
                f"{len(n_pts_tuple)} parametric dimensions."
            )
        node_list: list[npt.NDArray[np.floating[Any]]] = [
            np.asarray(a, dtype=dtype) for a in nodes.pts_per_dir
        ]
        for i, (arr_i, n) in enumerate(zip(node_list, n_pts_tuple, strict=True)):
            if arr_i.shape[0] != n:
                raise ValueError(
                    f"PointsLattice direction {i} has {arr_i.shape[0]} nodes, expected {n}."
                )
        return node_list

    # Single 1D array
    if isinstance(nodes, np.ndarray) and nodes.ndim == 1:
        if len(n_pts_tuple) != 1:
            raise ValueError(
                f"A single node array implies 1D fitting, but values have "
                f"{len(n_pts_tuple)} parametric dimensions."
            )
        arr: npt.NDArray[np.floating[Any]] = nodes.astype(dtype, copy=False)
        if arr.shape[0] != n_pts_tuple[0]:
            raise ValueError(
                f"Node array length {arr.shape[0]} does not match "
                f"values grid size {n_pts_tuple[0]}."
            )
        return [arr]

    # Sequence of arrays
    node_list_seq: list[npt.NDArray[np.floating[Any]]] = [np.asarray(a, dtype=dtype) for a in nodes]
    if len(node_list_seq) != len(n_pts_tuple):
        raise ValueError(f"Expected {len(n_pts_tuple)} node arrays, got {len(node_list_seq)}.")
    for i, (arr_i, n) in enumerate(zip(node_list_seq, n_pts_tuple, strict=True)):
        if arr_i.ndim != 1 or arr_i.shape[0] != n:
            raise ValueError(
                f"Node array for direction {i} has shape {arr_i.shape}, expected ({n},)."
            )
    return node_list_seq


def _is_scattered_nodes(
    nodes: (
        PointsLattice | npt.NDArray[np.floating[Any]] | Sequence[npt.NDArray[np.floating[Any]]]
    ),
) -> bool:
    """Check whether *nodes* represents scattered (non-tensor-product) points.

    Args:
        nodes: The nodes argument passed to :func:`fit_bezier`.

    Returns:
        bool: ``True`` if *nodes* is a 2D array (scattered points),
        ``False`` if it is a :class:`~pantr.quad.PointsLattice`, a 1D array,
        or a sequence of 1D arrays (tensor-product).
    """
    if isinstance(nodes, PointsLattice):
        return False
    return isinstance(nodes, np.ndarray) and nodes.ndim == 2  # noqa: PLR2004


def fit_bezier(  # noqa: PLR0912
    values: npt.ArrayLike,
    nodes: (
        PointsLattice | npt.NDArray[np.floating[Any]] | Sequence[npt.NDArray[np.floating[Any]]]
    ),
    *,
    degree: int | Sequence[int] | None = None,
    tol: float | None = None,
) -> Bezier:
    """Construct a Bézier from pre-evaluated sample values at known nodes.

    The output dtype is inferred from *values*.

    Supports two point layouts:

    - **Tensor-product** (a :class:`~pantr.quad.PointsLattice`, a single 1D
      array, or a sequence of 1D arrays): values must have shape
      ``(*n_pts_per_dir)`` (scalar) or ``(*n_pts_per_dir, rank)`` (vector).
      Per-direction SVD pseudo-inverse is used (efficient).
    - **Scattered** (a 2D ``ndarray`` of shape ``(n_pts, dim)``): values must
      have shape ``(n_pts,)`` (scalar) or ``(n_pts, rank)`` (vector).
      ``degree`` is required. A full tensor-product Vandermonde is built and
      solved via SVD.

    Args:
        values (npt.ArrayLike): Sample values.
        nodes: Interpolation nodes.

            - A :class:`~pantr.quad.PointsLattice`: tensor-product grid.
            - A 1D ``ndarray``: 1D tensor-product (single direction).
            - A sequence of 1D ``ndarray`` values: N-D tensor-product.
            - A 2D ``ndarray`` of shape ``(n_pts, dim)``: scattered points.
        degree (int | Sequence[int] | None): Polynomial degree per direction.
            Required for scattered nodes.  If *None* (default) for
            tensor-product nodes, ``degree = n_pts - 1`` (exact
            interpolation).  Must satisfy
            ``prod(degree + 1) <= n_pts`` for scattered nodes and
            ``degree < n_pts`` per direction for tensor-product.
        tol (float | None): SVD truncation tolerance. If *None*, uses a
            default based on machine epsilon.

    Returns:
        ~pantr.bezier.Bezier: A non-rational Bézier.

    Raises:
        ValueError: If *nodes* are inconsistent with *values*, *degree* is
            invalid, or *degree* is missing for scattered nodes; or ("SVD did not
            converge") if a node in a direction with more than one node, or a
            scattered point, is NaN or infinite.
    """
    from . import Bezier as BezierCls  # noqa: PLC0415

    values_untyped = np.asarray(values)
    if not np.issubdtype(values_untyped.dtype, np.floating):
        values_untyped = values_untyped.astype(np.float64)
    values_arr: npt.NDArray[np.floating[Any]] = values_untyped
    dtype_obj = values_arr.dtype

    # --- Scattered path ---
    if _is_scattered_nodes(nodes):
        pts: npt.NDArray[np.floating[Any]] = np.asarray(nodes, dtype=dtype_obj)
        ndim_param = pts.shape[1]
        n_pts = pts.shape[0]

        if degree is None:
            raise ValueError("degree is required for scattered (non-tensor-product) nodes.")

        degree_tuple = (degree,) * ndim_param if isinstance(degree, int) else tuple(degree)
        if len(degree_tuple) != ndim_param:
            raise ValueError(
                f"degree has {len(degree_tuple)} entries but nodes have {ndim_param} columns."
            )
        n_coeffs = int(np.prod(tuple(d + 1 for d in degree_tuple)))
        if n_coeffs > n_pts:
            raise ValueError(
                f"Underdetermined: {n_coeffs} coefficients but only {n_pts} sample points."
            )

        # Validate values shape
        if values_arr.ndim == 1:
            if values_arr.shape[0] != n_pts:
                raise ValueError(
                    f"values has {values_arr.shape[0]} entries but nodes has {n_pts} points."
                )
        elif values_arr.ndim == 2:  # noqa: PLR2004
            if values_arr.shape[0] != n_pts:
                raise ValueError(
                    f"values has {values_arr.shape[0]} rows but nodes has {n_pts} points."
                )
        else:
            raise ValueError(
                f"For scattered nodes, values must be 1D (scalar) or 2D "
                f"(vector), got {values_arr.ndim}D."
            )

        compute = _compute_dtype(dtype_obj)
        vals2d = values_arr if values_arr.ndim == 2 else values_arr[:, np.newaxis]  # noqa: PLR2004
        fit = _cpp.fit_scattered_f32 if compute == np.float32 else _cpp.fit_scattered
        return BezierCls._from_handle(
            fit(
                np.ascontiguousarray(vals2d, dtype=compute),
                np.ascontiguousarray(pts, dtype=compute),
                list(degree_tuple),
                tol,
            )
        )

    # --- Tensor-product path ---
    # Determine parametric dimension from nodes
    if isinstance(nodes, PointsLattice):
        ndim_param = nodes.dim
    elif isinstance(nodes, np.ndarray) and nodes.ndim == 1:
        ndim_param = 1
    else:
        ndim_param = len(nodes)

    # Infer n_pts from values shape: first ndim_param dimensions are the grid
    if values_arr.ndim == ndim_param:
        # Scalar-valued
        n_pts_tuple = values_arr.shape
    elif values_arr.ndim == ndim_param + 1:
        # Vector-valued
        n_pts_tuple = values_arr.shape[:ndim_param]
    else:
        raise ValueError(
            f"values has {values_arr.ndim} dimensions, expected {ndim_param} "
            f"(scalar) or {ndim_param + 1} (vector) for {ndim_param}D fitting."
        )

    degree_tuple_tp = _validate_degree(degree, n_pts_tuple)
    node_arrays = _resolve_nodes_from_user(nodes, n_pts_tuple, dtype_obj)
    return _fit_tensor(values_arr, node_arrays, degree_tuple_tp, tol, _compute_dtype(dtype_obj))
