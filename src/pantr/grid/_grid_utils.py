"""Shared Layer-2 helpers for the grid package.

Re-exports the ``float64`` coercion helper from :mod:`pantr.geometry`, so the
implementation lives in a single place.
"""

from __future__ import annotations

from ..geometry import _as_float64

__all__ = ["_as_float64"]
