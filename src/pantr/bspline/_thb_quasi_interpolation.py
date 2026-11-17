r"""Quasi-interpolation onto THB spline spaces (Speleers-Manni, effortless).

This module provides :func:`quasi_interpolate_thb_spline`, the hierarchical
counterpart of :func:`~pantr.bspline.quasi_interpolate_bspline`.  Following
Speleers-Manni (2016), the hierarchical quasi-interpolant is assembled with no
inter-level coupling: each active hierarchical dof receives the coefficient of a
per-level tensor-product quasi-interpolant,

.. math::
    Q_H f = \sum_{\ell} \sum_{\beta \in A_\ell} \lambda^\ell_\beta(f)\,
            \operatorname{trunc}(B^\ell_\beta),

so ``c[d] = λ^{l}_β(f)`` for the active function ``(l, β)`` of global dof ``d``
(truncation already lives in the basis).  The per-level functional is the
Lee-Lyche-Mørken local projector, evaluated on a **level-l active leaf cell** inside
``supp(B^l_β)``.  On such a cell, truncation zeros every coarser active function's
component on the active ``B^l_β``, so ``λ^l_β(s) = c_β`` for any THB spline ``s`` and
``Q_H`` reproduces the THB space exactly.

The points to sample and the contraction of the samples with the weights are the C++ library's
(``pantr::thb::THBQuasiInterpolation``, ``include/pantr/thb/quasi_interpolate.hpp``); this module
calls ``func`` on the points in between.

Main exports:

- :func:`quasi_interpolate_thb_spline`: quasi-interpolate a callable onto a
  :class:`~pantr.bspline.THBSplineSpace`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, get_args

import numpy as np
from numpy import typing as npt

from .._ext import ext
from ._bspline_quasi_interpolation import QIKind, _evaluate_func
from ._thb_spline import THBSpline
from ._thb_spline_space import THBSplineSpace

if TYPE_CHECKING:
    from collections.abc import Callable

_cpp: Any = ext.bspline
"""The compiled THB quasi-interpolants (private; untyped: one class per dtype)."""

_F32 = np.dtype(np.float32)


def quasi_interpolate_thb_spline(
    func: Callable[[npt.NDArray[np.float64]], npt.ArrayLike],
    space: THBSplineSpace,
    *,
    kind: QIKind = "llm",
) -> THBSpline:
    """Quasi-interpolate a callable onto a THB spline space.

    Assembles the Speleers-Manni hierarchical quasi-interpolant: every active dof gets
    the coefficient of the per-level Lee-Lyche-Mørken local projector, sampled on a
    level-``l`` active leaf cell inside the function's support.  For the truncated
    (THB) basis this reproduces any THB spline exactly; for the non-truncated (HB)
    basis it remains a valid local approximant but is not an exact projector.

    Args:
        func (Callable): Function to quasi-interpolate.  Called once on an
            ``(M, dim)`` point array and must return ``(M,)`` (scalar) or
            ``(M, rank)`` (vector-valued).
        space (THBSplineSpace): The target hierarchical space.
        kind (QIKind): Quasi-interpolant kind.  Only ``"llm"`` (Lee-Lyche-Mørken) is
            currently supported.  Defaults to ``"llm"``.

    Returns:
        THBSpline: A THB spline whose evaluation quasi-interpolates ``func``.

    Raises:
        TypeError: If ``space`` is not a :class:`~pantr.bspline.THBSplineSpace`.
        ValueError: If ``kind`` is not recognized, or if ``func`` returns an output
            with an invalid shape (0-D, more than 2-D, or wrong leading dimension).
        RuntimeError: If the grid has been modified since ``space`` was constructed,
            or an active dof has no leaf cell at its level (inconsistent space).

    References:
        Hierarchical quasi-interpolation in truncated hierarchical spaces
        :cite:p:`speleers2016quasi`.
    """
    if not isinstance(space, THBSplineSpace):
        raise TypeError(f"space must be a THBSplineSpace; got {type(space).__name__!r}.")
    if kind not in get_args(QIKind):
        valid = ", ".join(repr(v) for v in get_args(QIKind))
        raise ValueError(f"Unknown kind {kind!r}; expected one of {valid}.")
    space._check_not_stale()

    f32 = space._root_space.dtype == _F32
    qi = (_cpp.THBQuasiInterpolationF32 if f32 else _cpp.THBQuasiInterpolationF64)(space._h)
    # the local functionals are built before func is sampled
    values, _rank, scalar = _evaluate_func(func, qi.points())
    coeffs = np.asarray(qi.coefficients(np.ascontiguousarray(values)))
    if scalar:
        return THBSpline(space, coeffs[:, 0])
    return THBSpline(space, coeffs)
