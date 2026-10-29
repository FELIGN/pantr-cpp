"""Layer 2 of Bernstein root finding: validation, the active provider, and warnings.

Validates the Béziers, extracts their scalar coefficients, resolves the tolerance and calls the
active root provider (:func:`~pantr.bezier.get_root_provider`): pantr's own (``"builtin"``, the
compiled ``pantr._pantr.roots``) or BeRF-FR's (``"berf"``, the ``pantr_berf`` package). A provider
is a module with the functions ``find_roots``, ``find_roots_batch``, ``find_monotone_root`` and
``find_monotone_root_batch`` (each with an ``_f32`` twin for ``float32`` coefficients), taking
C-contiguous coefficient arrays and returning arrays; a result that does not account for the
whole interval comes with a completeness code, which becomes a :class:`RuntimeWarning` here.

This module is not part of the public API. Users should call the functions
exported from :mod:`pantr.bezier`.
"""

from __future__ import annotations

import importlib
import os
import types
import warnings
from collections.abc import Sequence
from typing import Any, Final

import numpy as np
from numpy import typing as npt

from .._ext import ext
from ..tolerance import get_strict
from ._bezier import Bezier

_BUILTIN: Final[str] = "builtin"
"""Name of pantr's own root provider."""

_BERF: Final[str] = "berf"
"""Name of the BeRF-FR root provider (the ``pantr-berf`` package)."""

_PROVIDERS: Final[tuple[str, ...]] = (_BUILTIN, _BERF)
"""The provider names :func:`set_root_provider` accepts."""

ROOT_PROVIDER_ENV: Final[str] = "PANTR_ROOT_PROVIDER"
"""Environment variable naming the initial root provider, read on first use."""

_BERF_MODULE: Final[str] = "pantr_berf"
"""Module that provides the ``"berf"`` root provider."""

_COMPLETE: Final[int] = int(ext.roots.COMPLETE)
"""Completeness code of a result that accounts for all of ``[0, 1]``."""

_active: str | None = None
"""The active provider's name, or ``None`` until first use resolves it."""


def _load_berf() -> types.ModuleType:
    """Import the BeRF-FR provider module.

    Returns:
        types.ModuleType: The ``pantr_berf`` module.

    Raises:
        ImportError: If ``pantr-berf`` is not installed.
    """
    try:
        return importlib.import_module(_BERF_MODULE)
    except ImportError as exc:
        msg = "The 'berf' root provider needs the pantr-berf package, which could not be imported."

        raise ImportError(msg) from exc


def _check_name(name: object) -> str:
    """Validate a provider name.

    Args:
        name (object): The name to check.

    Returns:
        str: The name.

    Raises:
        TypeError: If ``name`` is not a string.
        ValueError: If ``name`` is not a known provider.
    """
    if not isinstance(name, str):
        msg = f"A root provider name must be a str, got {type(name).__name__}"
        raise TypeError(msg)
    if name not in _PROVIDERS:
        msg = f"Unknown root provider {name!r}; expected one of 'builtin', 'berf'"
        raise ValueError(msg)
    return name


def set_root_provider(name: str) -> None:
    """Select the provider that :func:`find_roots` and :func:`find_monotone_root` use.

    The choice is process-wide. Its initial value comes from the ``PANTR_ROOT_PROVIDER``
    environment variable, read on first use; without it, ``"berf"`` when the ``pantr-berf``
    package is installed, else ``"builtin"``.

    Args:
        name (str): ``"builtin"`` (pantr's own: Yuksel's algorithm and Bézier clipping) or
            ``"berf"`` (BeRF-FR, from the ``pantr-berf`` package).

    Raises:
        TypeError: If ``name`` is not a string.
        ValueError: If ``name`` is not a known provider.
        ImportError: If ``name`` is ``"berf"`` and ``pantr-berf`` cannot be imported.

    Example:
        >>> from pantr.bezier import get_root_provider, set_root_provider
        >>> previous = get_root_provider()
        >>> set_root_provider("builtin")
        >>> get_root_provider()
        'builtin'
        >>> set_root_provider(previous)
    """
    global _active  # noqa: PLW0603
    checked = _check_name(name)
    if checked == _BERF:
        _load_berf()
    _active = checked


def _default_provider() -> str:
    """Resolve the initial provider: the environment variable, else the installed default.

    Returns:
        str: The provider name.

    Raises:
        ValueError: If ``PANTR_ROOT_PROVIDER`` names no provider.
        ImportError: If it names ``"berf"`` and ``pantr-berf`` is not installed.
    """
    requested = os.environ.get(ROOT_PROVIDER_ENV, "")
    if requested:
        if requested not in _PROVIDERS:
            msg = (
                f"{ROOT_PROVIDER_ENV}={requested!r} names no root provider; "
                "expected one of 'builtin', 'berf'"
            )
            raise ValueError(msg)
        if requested == _BERF:
            _load_berf()
        return requested
    try:
        importlib.import_module(_BERF_MODULE)
    except ModuleNotFoundError as exc:
        # Only "pantr_berf is not installed" selects the builtin provider; an installed package
        # that fails to import (a missing library, an interface mismatch) is an error to see.
        if exc.name != _BERF_MODULE:
            raise
        return _BUILTIN
    return _BERF


def get_root_provider() -> str:
    """Get the name of the active root provider.

    Returns:
        str: ``"builtin"`` or ``"berf"``.

    Raises:
        ValueError: If no provider was set and ``PANTR_ROOT_PROVIDER`` names no provider.
        ImportError: If no provider was set, ``PANTR_ROOT_PROVIDER`` is ``"berf"``, and
            ``pantr-berf`` is not installed.
    """
    global _active  # noqa: PLW0603
    if _active is None:
        resolved = _default_provider()
        # Resolution may import pantr_berf: a set_root_provider from another thread meanwhile wins.
        if _active is None:
            _active = resolved
    return _active


def _get_state() -> str | None:
    """Get the raw provider state: the active name, or ``None`` before first use.

    Returns:
        str | None: The state (for tests, through ``pantr._testing``).
    """
    return _active


def _set_state(state: str | None) -> None:
    """Set the raw provider state, ``None`` meaning "resolve on first use" (for tests).

    Args:
        state (str | None): The state to restore.
    """
    global _active  # noqa: PLW0603
    _active = state


def _provider() -> Any:  # noqa: ANN401
    """Get the active provider's module (untyped: a compiled module or ``pantr_berf``).

    Returns:
        Any: The module.
    """
    if get_root_provider() == _BERF:
        return _load_berf()
    return ext.roots


def _function(name: str, dtype: np.dtype[Any]) -> Any:  # noqa: ANN401
    """Get a provider function for a coefficient dtype.

    Args:
        name (str): The function's name.
        dtype (np.dtype[Any]): ``float32`` or ``float64``.

    Returns:
        Any: The function.
    """
    return getattr(_provider(), name + ("_f32" if dtype == np.float32 else ""))


def _warn_incomplete(n_incomplete: int, n_total: int) -> None:
    """Warn that some results do not account for the whole interval.

    Args:
        n_incomplete (int): Polynomials whose result has a gap.
        n_total (int): Polynomials solved.
    """
    warnings.warn(
        f"{n_incomplete} of {n_total} polynomials exhausted the root finder's subdivision "
        "budget on part of [0, 1] before narrowing it to tol; a root there is reported only "
        "if its residual allows, so the result may be missing a root or be less accurate than "
        "tol asks for",
        RuntimeWarning,
        stacklevel=4,
    )


def _extract_coeff(bezier: Bezier) -> npt.NDArray[np.float32 | np.float64]:
    """Extract the 1-D coefficient array of a scalar Bézier curve.

    For a rational Bézier this is the first homogeneous component (the numerator ``x * w``),
    whose roots are those of ``x`` where the weights are positive.

    Args:
        bezier (Bezier): A validated 1-D scalar Bézier curve.

    Returns:
        npt.NDArray[np.float32 | np.float64]: Contiguous, length ``degree + 1``.
    """
    return np.ascontiguousarray(bezier.control_points[:, 0])


def _validate_bezier_for_roots(bezier: object) -> Bezier:
    """Validate that an object is a scalar 1-D Bézier.

    Args:
        bezier (object): Input to validate.

    Returns:
        Bezier: The validated Bézier.

    Raises:
        TypeError: If ``bezier`` is not a :class:`Bezier`.
        ValueError: If ``bezier.dim != 1`` or ``bezier.rank != 1``.
    """
    if not isinstance(bezier, Bezier):
        msg = f"Expected a Bezier instance, got {type(bezier).__name__}"
        raise TypeError(msg)
    if bezier.dim != 1:
        msg = f"Bezier must be 1-D (dim == 1), got dim={bezier.dim}"
        raise ValueError(msg)
    if bezier.rank != 1:
        msg = f"Bezier must be scalar (rank == 1), got rank={bezier.rank}"
        raise ValueError(msg)
    return bezier


def _extract_batch_coeffs(
    beziers: Sequence[Bezier],
) -> npt.NDArray[np.float32 | np.float64]:
    """Validate a sequence of Béziers and stack their coefficients.

    Args:
        beziers (Sequence[Bezier]): Béziers of one degree.

    Returns:
        npt.NDArray[np.float32 | np.float64]: ``(n_polys, degree + 1)``, the first Bézier's dtype
        (``(0, 1)`` ``float64`` for an empty sequence).

    Raises:
        TypeError: If any element is not a :class:`Bezier`.
        ValueError: If any Bézier has ``dim != 1`` or ``rank != 1``, or the degrees differ.
    """
    n_polys = len(beziers)
    if n_polys == 0:
        return np.empty((0, 1), dtype=np.float64)

    first = _validate_bezier_for_roots(beziers[0])
    degree = first.degree[0]
    coeffs = np.empty((n_polys, degree + 1), dtype=first.dtype)
    coeffs[0] = _extract_coeff(first)
    for i in range(1, n_polys):
        bez = _validate_bezier_for_roots(beziers[i])
        if bez.degree[0] != degree:
            msg = (
                f"All Beziers must have the same degree. "
                f"Bezier 0 has degree {degree}, Bezier {i} has degree {bez.degree[0]}"
            )
            raise ValueError(msg)
        coeffs[i] = _extract_coeff(bez)
    return coeffs


def _resolve_tol(dtype: np.dtype[Any], tol: float | None) -> float:
    """Resolve the tolerance: the user's, or ``get_strict`` of the coefficients' dtype.

    Args:
        dtype (np.dtype[Any]): The coefficients' dtype.
        tol (float | None): The user's tolerance, or ``None``.

    Returns:
        float: The tolerance.

    Raises:
        ValueError: If ``tol`` is not positive.
    """
    if tol is None:
        return get_strict(dtype)
    if tol <= 0.0:
        msg = f"tol must be positive, got {tol}"
        raise ValueError(msg)
    return tol


def _find_roots_impl(bezier: Bezier, *, tol: float | None = None) -> npt.NDArray[np.float64]:
    """L2 implementation of :func:`pantr.bezier.find_roots` for one Bézier."""
    coeff = _extract_coeff(_validate_bezier_for_roots(bezier))
    resolved = float(_resolve_tol(coeff.dtype, tol))
    roots, completeness = _function("find_roots", coeff.dtype)(coeff, resolved, resolved)
    if completeness != _COMPLETE:
        _warn_incomplete(1, 1)
    return np.asarray(roots, dtype=np.float64)


def _find_roots_batch_impl(
    beziers: Sequence[Bezier], *, tol: float | None = None
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.intp]]:
    """L2 implementation of :func:`pantr.bezier.find_roots` for a sequence."""
    coeffs = _extract_batch_coeffs(beziers)
    n_polys, n_coeff = coeffs.shape
    if n_polys == 0:
        return np.empty((0, max(n_coeff - 1, 1)), dtype=np.float64), np.empty(0, dtype=np.intp)
    resolved = float(_resolve_tol(coeffs.dtype, tol))
    roots, counts, completeness = _function("find_roots_batch", coeffs.dtype)(
        coeffs, resolved, resolved
    )
    n_incomplete = int(np.count_nonzero(completeness != _COMPLETE))
    if n_incomplete:
        _warn_incomplete(n_incomplete, n_polys)
    return np.asarray(roots, dtype=np.float64), np.asarray(counts, dtype=np.intp)


def _solve_monotone_root_impl(bezier: Bezier, *, tol: float | None = None) -> float:
    """L2 implementation of :func:`pantr.bezier.find_monotone_root` for one Bézier."""
    coeff = _extract_coeff(_validate_bezier_for_roots(bezier))
    resolved = float(_resolve_tol(coeff.dtype, tol))
    return float(_function("find_monotone_root", coeff.dtype)(coeff, resolved))


def _solve_monotone_root_batch_impl(
    beziers: Sequence[Bezier], *, tol: float | None = None
) -> npt.NDArray[np.float64]:
    """L2 implementation of :func:`pantr.bezier.find_monotone_root` for a sequence."""
    coeffs = _extract_batch_coeffs(beziers)
    if coeffs.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    resolved = float(_resolve_tol(coeffs.dtype, tol))
    return np.asarray(
        _function("find_monotone_root_batch", coeffs.dtype)(coeffs, resolved), dtype=np.float64
    )
