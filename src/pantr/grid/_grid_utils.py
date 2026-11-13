"""Shared Layer-2 helpers for the grid package.

Re-exports the ``float64`` coercion helper from :mod:`pantr.geometry`, so the
implementation lives in a single place, and reads the state a previous release pickled.
"""

from __future__ import annotations

from typing import Any

from ..geometry import _as_float64


def _previous_release_slots(state: Any) -> dict[str, Any]:  # noqa: ANN401
    """Return the slot name to value map of an object a previous release pickled.

    A previous release gave its grid classes ``__slots__`` and no custom pickling, so the
    pickle holds the state ``object.__reduce_ex__`` gives for such a class.

    Args:
        state (Any): ``(dict_state, slots_state)``, either of which may be ``None``; or a
            bare mapping.

    Returns:
        dict[str, Any]: The attribute and slot values, by name.
    """
    if isinstance(state, tuple):
        merged: dict[str, Any] = {}
        for part in state:
            if part is not None:
                merged.update(part)
        return merged
    return dict(state)


__all__ = ["_as_float64", "_previous_release_slots"]
