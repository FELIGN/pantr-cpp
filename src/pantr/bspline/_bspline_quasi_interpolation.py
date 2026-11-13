"""Quasi-interpolation onto tensor-product B-spline spaces (Lee-Lyche-Mørken).

This module provides :func:`quasi_interpolate_bspline`, a local quasi-interpolant
that maps a callable ``f`` to a :class:`~pantr.bspline.Bspline`.  It uses the
Lee-Lyche-Mørken local-spline-interpolation construction: for each basis function
``B_β`` the coefficient is a local, point-value functional
``λ_β(f) = Σ_k a_{β,k} f(x_{β,k})`` obtained by inverting a small collocation
matrix on one knot-interval cell inside ``supp(B_β)``.  On a single cell the
``p+1`` (per direction) nonzero B-splines span the degree-``p`` polynomials, so the
local interpolation recovers their exact coefficients and the operator is a
projector onto the spline space (``Q B_β = B_β``).

The tensor-product operator is built and applied in C++
(``include/pantr/bspline/quasi_interpolate.hpp``): it lists the points to sample, this module
calls ``f`` on them, and C++ contracts the samples with the weights. :data:`QIKind` and
:func:`_evaluate_func` are shared with the hierarchical quasi-interpolant in
``_thb_quasi_interpolation``, which is staged the same way.

Main exports:

- :func:`quasi_interpolate_bspline`: quasi-interpolate a callable onto a
  tensor-product :class:`~pantr.bspline.BsplineSpace`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, get_args

import numpy as np
from numpy import typing as npt

from .._ext import ext
from ._bspline import Bspline
from ._bspline_space_nd import BsplineSpace

_cpp: Any = ext.bspline
"""The compiled quasi-interpolants (private; untyped: one class per dtype)."""

if TYPE_CHECKING:
    from collections.abc import Callable

QIKind = Literal["llm"]
"""Supported quasi-interpolation kinds (Lee-Lyche-Mørken local projector)."""


# ---------------------------------------------------------------------------
# Batched evaluation helper
# ---------------------------------------------------------------------------


def _evaluate_func(
    func: Callable[[npt.NDArray[np.float64]], npt.ArrayLike],
    points: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], int, bool]:
    """Evaluate ``func`` on a point array and normalize the output.

    Args:
        func (Callable): Called on ``(M, dim)`` points; returns ``(M,)`` or
            ``(M, rank)``.
        points (npt.NDArray[np.float64]): Points of shape ``(M, dim)``.

    Returns:
        tuple[npt.NDArray[np.float64], int, bool]: ``(values, rank, scalar)`` with
        ``values`` of shape ``(M, rank)``.

    Raises:
        ValueError: If the output is 0-D, has more than 2 dimensions, or its leading
            dimension does not match ``M``.
    """
    values = np.asarray(func(np.ascontiguousarray(points, dtype=np.float64)), dtype=np.float64)
    if values.ndim == 0:
        raise ValueError(
            f"func returned a 0-D scalar; expected shape ({points.shape[0]},) "
            f"or ({points.shape[0]}, rank)."
        )
    if values.ndim > 2:  # noqa: PLR2004
        raise ValueError(
            f"func returned a {values.ndim}-D array of shape {values.shape!r}; "
            f"expected shape ({points.shape[0]},) or ({points.shape[0]}, rank)."
        )
    if values.shape[0] != points.shape[0]:
        raise ValueError(f"func returned {values.shape[0]} values for {points.shape[0]} points.")
    if values.ndim == 1:
        return values.reshape(-1, 1), 1, True
    return values, int(values.shape[1]), False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def quasi_interpolate_bspline(
    func: Callable[[npt.NDArray[np.float64]], npt.ArrayLike],
    space: BsplineSpace,
    *,
    kind: QIKind = "llm",
) -> Bspline:
    """Quasi-interpolate a callable onto a tensor-product B-spline space.

    Builds the Lee-Lyche-Mørken local projector: a local, point-value functional per
    basis function whose weights come from inverting a small collocation block.  The
    operator reproduces the spline space (``Q B_β = B_β``) and is local.

    Args:
        func (Callable): Function to quasi-interpolate.  Called once on an
            ``(M, dim)`` point array and must return ``(M,)`` (scalar) or
            ``(M, rank)`` (vector-valued).  Unlike the lattice-based
            :func:`interpolate_bspline` / :func:`l2_project_bspline`, ``func``
            receives a flat point array because the QI samples scattered local points.
        space (BsplineSpace): The target tensor-product B-spline space.
        kind (QIKind): Quasi-interpolant kind.  Only ``"llm"`` (Lee-Lyche-Mørken) is
            currently supported.  Defaults to ``"llm"``.

    Returns:
        Bspline: A non-rational B-spline whose evaluation quasi-interpolates ``func``.

    Raises:
        TypeError: If ``space`` is not a :class:`~pantr.bspline.BsplineSpace`.
        ValueError: If ``kind`` is not recognized, ``space`` has no direction, or
            ``func`` returns an output with an invalid shape (0-D, more than 2-D, or
            wrong leading dimension).

    Note:
        All internal computation uses ``float64``.  The returned control points are
        cast to ``space.dtype`` at the end, so a ``float32`` space incurs a precision
        loss on the final cast.

    Example:
        >>> import numpy as np
        >>> from pantr.bspline import create_uniform_space, quasi_interpolate_bspline
        >>> space = create_uniform_space([2], [4])
        >>> qi = quasi_interpolate_bspline(lambda p: p[:, 0] ** 2, space)
        >>> bool(np.isclose(qi.evaluate(np.array([[0.3]])), 0.09))
        True
    """
    if not isinstance(space, BsplineSpace):
        raise TypeError(f"space must be a BsplineSpace; got {type(space).__name__!r}.")
    if kind not in get_args(QIKind):
        valid = ", ".join(repr(v) for v in get_args(QIKind))
        raise ValueError(f"Unknown kind {kind!r}; expected one of {valid}.")
    if space.dim == 0:
        # The previous release's message (NumPy's, from stacking no direction's points).
        raise ValueError("need at least one array to stack")

    cls = _cpp.QuasiInterpolationF32 if space.dtype == np.float32 else _cpp.QuasiInterpolationF64
    qi = cls(space._h)  # the local functionals, built before func is sampled
    values, _rank, _scalar = _evaluate_func(func, qi.points())
    coeffs = np.asarray(qi.coefficients(np.ascontiguousarray(values)))
    control_points = coeffs.astype(space.dtype, copy=False)
    return Bspline(space, control_points)
