"""Argument handling of the B-spline refinement methods that moved into the compiled library.

``insert_knots``, ``subdivide``, ``restrict``, ``split`` and ``to_periodic`` normalise their
arguments in Python and validate values in the compiled library. These tests pin what that split
does with the arguments an array library throws at a method: NumPy scalars of other widths, masked
and 0-d arrays, booleans for integers, integers beyond 64 bits, and the values that are not
numbers. The oracle is the behavior of the pure-Python implementation these methods replaced,
measured on it case by case; the places that differ on purpose say so.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from pantr.bspline import (
    Bspline,
    BsplineSpace,
    BsplineSpace1D,
    create_uniform_open_knots,
    create_uniform_periodic_knots,
)


def _curve(dtype: type = np.float64) -> Bspline:
    """Build a quadratic curve on four intervals of ``[0, 1]``."""
    knots: npt.NDArray[Any] = create_uniform_open_knots(4, 2).astype(dtype)
    space = BsplineSpace([BsplineSpace1D(knots, 2)])
    cp: npt.NDArray[Any] = (np.arange(12).reshape(6, 2) / 8.0 - 0.5).astype(dtype)
    return Bspline(space, cp)


def _periodic_open() -> Bspline:
    """Build the open form of a periodic quadratic curve."""
    knots = create_uniform_periodic_knots(6, 2)
    space = BsplineSpace([BsplineSpace1D(knots, 2, periodic=True)])
    cp = np.random.default_rng(3).integers(-8, 9, (space.num_basis[0], 2)) / 8.0
    return Bspline(space, cp).to_open_bspline()


class TestInsertKnotsArguments:
    """What ``insert_knots`` accepts as the knots to insert."""

    def test_a_masked_array_inserts_every_value(self) -> None:
        """The mask is dropped by ``np.asarray``, as it always was."""
        out = _curve().insert_knots(np.ma.masked_array([0.3, 0.6], mask=[False, True]))
        assert out.space.spaces[0].num_basis == 8

    def test_big_endian_and_integer_arrays_are_converted(self) -> None:
        """Other dtypes are converted to the spline's, not refused."""
        assert _curve().insert_knots(np.array([0.3], dtype=">f8")).space.spaces[0].num_basis == 7
        wide = Bspline(
            BsplineSpace([BsplineSpace1D(4.0 * create_uniform_open_knots(4, 2), 2)]),
            np.arange(12.0).reshape(6, 2),
        )
        assert wide.insert_knots(np.array([1, 2])).space.spaces[0].num_basis == 8

    def test_a_zero_dimensional_array_is_not_one_dimensional(self) -> None:
        """A 0-d array has size one and shape ``()``: refused with that shape."""
        with pytest.raises(ValueError, match=r"new_knots must be a 1D array-like, got shape \(\)"):
            _curve().insert_knots(np.array(0.5))

    def test_a_two_dimensional_array_is_refused_with_its_shape(self) -> None:
        """The shape is in the message."""
        with pytest.raises(ValueError, match=r"got shape \(1, 2\)"):
            _curve().insert_knots([[0.1, 0.2]])

    def test_an_empty_array_of_any_shape_inserts_nothing_and_is_refused(self) -> None:
        """Size zero is "no knots" whatever the shape."""
        with pytest.raises(ValueError, match="At least one direction"):
            _curve().insert_knots(np.empty((0, 3)))

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_non_finite_values_are_outside_the_domain(self, bad: float) -> None:
        """NaN and infinities are not in the domain, and the message lists them."""
        with pytest.raises(ValueError, match=r"outside the domain \[0\.0, 1\.0\]: \["):
            _curve().insert_knots([bad])

    def test_longdouble_knots_are_cast_to_the_spline_dtype(self) -> None:
        """Extended precision values are rounded to the spline's precision."""
        out = _curve(np.float32).insert_knots(np.array([0.1], dtype=np.longdouble))
        assert out.dtype == np.float32
        assert out.space.spaces[0].knots[3] == np.float32(0.1)


class TestRestrictArguments:
    """What ``restrict`` accepts as bounds."""

    @pytest.mark.parametrize("scalar", [np.float16, np.float32, np.float64, np.longdouble])
    def test_numpy_scalar_bounds_are_accepted(self, scalar: type) -> None:
        """Bounds of any NumPy float width work.

        The pure-Python implementation refused ``np.float16`` and ``np.float32`` bounds with an
        unrelated ``ValueError`` ("knots must have at least 2*degree+2 elements"): their
        arithmetic stayed in the narrow type and the cut fell in the wrong place. Bounds are now
        read as Python floats, which is the behavior ``np.float64`` and ``float`` always had.
        """
        lo, hi = scalar(0.25), scalar(0.75)
        out = _curve().restrict((lo, hi))
        assert tuple(map(float, out.space.spaces[0].domain)) == (0.25, 0.75)

    def test_integer_bounds_are_accepted(self) -> None:
        """``(0, 1)`` is the full domain and ``(0, 0.5)`` a half."""
        out = _curve().restrict((0, 0.5))
        assert tuple(map(float, out.space.spaces[0].domain)) == (0.0, 0.5)
        with pytest.raises(ValueError, match="already open"):
            _curve().restrict((0, 1))

    @pytest.mark.parametrize("bounds", [("a", "b"), ("0.1", "0.5"), (None, 0.5)])
    def test_bounds_that_are_not_numbers_are_a_type_error(
        self, bounds: tuple[object, object]
    ) -> None:
        """A string is not parsed as a number."""
        with pytest.raises(TypeError):
            _curve().restrict(bounds)  # type: ignore[arg-type]

    def test_nan_bounds_are_refused_as_outside_the_domain(self) -> None:
        """A NaN bound is not a point of the domain."""
        with pytest.raises(ValueError, match="outside the domain"):
            _curve().restrict((np.nan, 0.5))

    def test_the_lower_bound_may_sit_within_the_tolerance_below_the_start(self) -> None:
        """A bound within the knot tolerance of an end is that end."""
        out = _curve().restrict((-1e-17, 0.5))
        assert float(out.space.spaces[0].domain[0]) == 0.0


class TestSplitArguments:
    """What ``split`` accepts as direction and value."""

    @pytest.mark.parametrize("direction", [False, np.int64(0)])
    def test_boolean_and_numpy_integer_directions_are_indices(self, direction: object) -> None:
        """Anything with ``__index__`` selects a direction."""
        left, right = _curve().split(direction, 0.3)  # type: ignore[arg-type]
        assert float(left.space.spaces[0].domain[1]) == 0.3
        assert float(right.space.spaces[0].domain[0]) == 0.3

    def test_a_float_direction_is_a_type_error(self) -> None:
        """``0.0`` is not an index."""
        with pytest.raises(TypeError):
            _curve().split(0.0, 0.3)  # type: ignore[arg-type]

    def test_a_direction_beyond_64_bits_is_out_of_range(self) -> None:
        """A huge integer is an out-of-range direction, not a conversion error.

        The message shows the integer saturated to 64 bits.
        """
        with pytest.raises(ValueError, match=r"direction must be in \[0, 1\)"):
            _curve().split(10**30, 0.3)

    @pytest.mark.parametrize("scalar", [np.float16, np.float32, np.longdouble])
    def test_numpy_scalar_values_are_accepted(self, scalar: type) -> None:
        """A split value of any NumPy float width works."""
        left, right = _curve().split(0, scalar(0.25))
        assert float(left.space.spaces[0].domain[1]) == 0.25
        assert float(right.space.spaces[0].domain[0]) == 0.25

    def test_a_string_value_is_a_type_error(self) -> None:
        """A string is not parsed as a number."""
        with pytest.raises(TypeError):
            _curve().split(0, "a")  # type: ignore[arg-type]

    def test_an_integer_value_is_spelled_as_a_float_in_the_message(self) -> None:
        """The compiled message spells the value as a Python float (``1.0``, not ``1``)."""
        with pytest.raises(ValueError, match=r"got 1\.0\."):
            _curve().split(0, 1)


class TestSubdivideArguments:
    """What ``subdivide`` accepts as counts and regularity."""

    def test_a_boolean_count_is_an_integer(self) -> None:
        """``True`` is a count of one: nothing to subdivide."""
        with pytest.raises(ValueError, match="At least one direction"):
            _curve().subdivide(True)

    def test_a_numpy_integer_count_is_not_a_sequence(self) -> None:
        """Only a Python ``int`` stands for all directions."""
        with pytest.raises(TypeError):
            _curve().subdivide(np.int64(2))  # type: ignore[arg-type]

    def test_a_float_regularity_is_a_type_error(self) -> None:
        """The regularity is an integer.

        The pure-Python implementation took ``1.0`` by accident (``degree - 1.0`` happens to
        work in the one place it is used); a float regularity is now refused.
        """
        with pytest.raises(TypeError):
            _curve().subdivide(2, 1.0)  # type: ignore[arg-type]

    def test_a_regularity_beyond_64_bits_is_out_of_range(self) -> None:
        """A huge regularity is out of ``[-1, degree - 1]``, as any large one is."""
        with pytest.raises(ValueError, match=r"regularity must be in \[-1, degree - 1\]"):
            _curve().subdivide(2, 10**30)

    def test_a_count_too_large_to_hold_the_knots_is_a_value_error(self) -> None:
        """No array could hold the knots; the request is refused, not attempted."""
        with pytest.raises(ValueError):
            _curve().subdivide(10**30)


class TestToPeriodicArguments:
    """What ``to_periodic`` accepts as continuity."""

    def test_a_boolean_is_an_integer_continuity(self) -> None:
        """``True`` is continuity one."""
        out = _periodic_open().to_periodic(True)
        assert out.space.spaces[0].periodic

    def test_a_numpy_integer_in_a_tuple_is_an_integer_continuity(self) -> None:
        """Tuple entries use ``__index__``."""
        out = _periodic_open().to_periodic((np.int64(1),))  # type: ignore[arg-type]
        assert out.space.spaces[0].periodic

    def test_minus_one_is_the_maximum_regularity(self) -> None:
        """``-1`` and ``None`` both mean ``degree - 1``."""
        a = _periodic_open().to_periodic(-1)
        b = _periodic_open().to_periodic()
        np.testing.assert_array_equal(a.control_points, b.control_points)

    def test_a_float_continuity_is_a_type_error(self) -> None:
        """A float is neither an integer nor a sequence."""
        with pytest.raises(TypeError):
            _periodic_open().to_periodic(1.0)  # type: ignore[arg-type]

    def test_a_continuity_beyond_64_bits_is_out_of_range(self) -> None:
        """A huge continuity is out of ``[0, degree - 1]``."""
        with pytest.raises(ValueError, match=r"continuity must be in \[0, degree-1\]=\[0, 1\]"):
            _periodic_open().to_periodic(10**30)

    def test_a_tuple_of_the_wrong_length_is_refused(self) -> None:
        """One entry per direction."""
        with pytest.raises(ValueError, match=r"continuity tuple length 2 != dimension 1"):
            _periodic_open().to_periodic((1, 1))
