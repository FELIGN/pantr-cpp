"""The binding's array helpers: views of empty storage stay read-only."""

from __future__ import annotations

import numpy as np
import pytest

from pantr._testing import empty_views


def test_empty_views_are_read_only() -> None:
    """A view over null data (an empty vector's) is read-only, empty, and of its own dtype.

    NumPy answers a null buffer with a fresh writable array, so the helper must hand it a valid
    address instead; a writable result here is that defect.
    """
    doubles, ints = empty_views()
    assert doubles.shape == (0,)
    assert ints.shape == (0,)
    assert doubles.dtype == np.float64
    assert ints.dtype.type is np.int64
    for view in (doubles, ints):
        assert not view.flags.writeable
        with pytest.raises(ValueError, match="WRITEABLE"):
            view.setflags(write=True)
