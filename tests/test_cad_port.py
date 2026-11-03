"""Behavior of ``pantr.cad`` on edge inputs and mixed precision, as the compiled library runs it.

The functions of ``pantr.cad`` that run in C++ keep the behavior of the pure-Python implementation
they replaced, probed on it input by input. Each test here pins one of those rows. A comment
"inherited from main" marks a behavior kept as it was and not chosen: where the Python
implementation refused or accepted an input for a reason of its own (a NumPy broadcast, a mix of
dtypes met late), the compiled library does the same, and the row says so. Where the library's
refusal replaces a NumPy error whose text was NumPy's, only the exception type is pinned.

Mixed-precision rows pin the dtype of the result (and, for the refusals, the message that is
the Python implementation's own).
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from numpy import typing as npt
from numpy.testing import assert_allclose

from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D
from pantr.cad import (
    create_bilinear,
    create_circle,
    create_coons_surface,
    create_coons_volume,
    create_extrusion,
    create_line,
    create_revolution,
    create_ruled,
    create_sweep,
    create_trilinear,
    join,
    make_compat,
)

_DTYPE_MESSAGE = "All B-spline spaces must have the same data type."


def _curve(
    cp: npt.ArrayLike,
    *,
    degree: int = 1,
    knots: list[float] | None = None,
    dtype: npt.DTypeLike = np.float64,
    rational: bool = False,
) -> Bspline:
    """Build a curve with clamped knots (``[0, 0, 1, 1]`` for degree 1 by default)."""
    k = np.array(
        [0.0] * (degree + 1) + [1.0] * (degree + 1) if knots is None else knots, dtype=dtype
    )
    return Bspline(
        BsplineSpace([BsplineSpace1D(k, degree)]), np.asarray(cp, dtype=dtype), is_rational=rational
    )


def _f32(b: Bspline) -> Bspline:
    """Return ``b`` with float32 knots and control points."""
    spaces = [
        BsplineSpace1D(np.asarray(s.knots, dtype=np.float32), s.degree, periodic=s.periodic)
        for s in b.space.spaces
    ]
    return Bspline(
        BsplineSpace(spaces),
        np.asarray(b.control_points, dtype=np.float32),
        is_rational=b.is_rational,
    )


def _surface(dtype: npt.DTypeLike = np.float64, *, x0: float = 0.0, shift: float = 0.0) -> Bspline:
    """A bilinear-in-u, degree-1 surface over ``[x0, x0 + 1] x [0, 1]`` with three v-points."""
    space = BsplineSpace(
        [
            BsplineSpace1D(np.array([x0, x0, x0 + 1, x0 + 1], dtype=dtype), 1),
            BsplineSpace1D(np.array([0, 0, 0.5, 1, 1], dtype=dtype), 1),
        ]
    )
    cp = np.zeros((2, 3, 3), dtype=dtype)
    for i in range(2):
        for j in range(3):
            cp[i, j] = [x0 + i, 0.5 * j, 0.0]
    cp[0] += shift
    return Bspline(space, cp)


def _random_surface(seed: int, dtype: npt.DTypeLike = np.float64) -> Bspline:
    rng = np.random.default_rng(seed)
    space = BsplineSpace(
        [
            BsplineSpace1D(np.array([0, 0, 0, 0.4, 1, 1, 1], dtype=dtype), 2),
            BsplineSpace1D(np.array([0, 0, 1, 1], dtype=dtype), 1),
        ]
    )
    return Bspline(space, rng.uniform(-1, 1, (*space.num_basis, 3)).astype(dtype))


def _boundary(b: Bspline, axis: int, side: int) -> Bspline:
    """Get the boundary of a surface or a volume (a B-spline, not a point)."""
    out = b.boundary(axis, side)
    assert isinstance(out, Bspline)
    return out


def _coons_curves(s: Bspline) -> tuple[tuple[Bspline, Bspline], tuple[Bspline, Bspline]]:
    """The boundary curves of a surface in the layout ``((C_v0, C_v1), (C_u0, C_u1))``."""
    return (
        (_boundary(s, 0, 0), _boundary(s, 0, 1)),
        (_boundary(s, 1, 0), _boundary(s, 1, 1)),
    )


def _volume(dtype: npt.DTypeLike = np.float64) -> Bspline:
    rng = np.random.default_rng(3)
    k = np.array([0, 0, 1, 1], dtype=dtype)
    space = BsplineSpace([BsplineSpace1D(k, 1) for _ in range(3)])
    return Bspline(space, rng.uniform(-1, 1, (2, 2, 2, 3)).astype(dtype))


def _faces(
    v: Bspline,
) -> tuple[tuple[Bspline, Bspline], tuple[Bspline, Bspline], tuple[Bspline, Bspline]]:
    """The six boundary faces of a volume in the layout of ``create_coons_volume``."""
    return (
        (_boundary(v, 0, 0), _boundary(v, 0, 1)),
        (_boundary(v, 1, 0), _boundary(v, 1, 1)),
        (_boundary(v, 2, 0), _boundary(v, 2, 1)),
    )


class TestEdgeInputs:
    """Accepted and refused edge inputs of the functions that run in C++."""

    def test_line_with_four_coordinates_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"Point must have at most 3 coordinates, got 4\."):
            create_line([0, 0, 0, 0])

    def test_circle_nan_angle_is_refused_by_the_rotation(self) -> None:
        with pytest.raises(ValueError, match=r"angle must be finite, got nan\."):
            create_circle(angle=float("nan"))

    def test_circle_zero_angle_is_one_span_of_coincident_points(self) -> None:
        arc = create_circle(angle=0.0)
        assert arc.space.num_basis == (3,)
        cp = np.asarray(arc.control_points)
        assert_allclose(cp, np.broadcast_to(cp[0], cp.shape), rtol=0, atol=0)

    def test_circle_angle_beyond_a_turn_is_four_spans(self) -> None:
        # ten radians is not wrapped: the span count is the number of quadrant bounds below it.
        assert create_circle(angle=10.0).space.num_basis == (9,)

    def test_bilinear_of_rank_zero_corners_is_all_zero_rank_three(self) -> None:
        srf = create_bilinear(np.zeros((2, 2, 0)))
        assert srf.rank == 3
        assert_allclose(np.asarray(srf.control_points), 0.0, rtol=0, atol=0)

    def test_revolution_axis_minus_one_is_the_last_coordinate_axis(self) -> None:
        crv = create_line((1, 0, 0), (1, 0, 1))
        a = create_revolution(crv, (0, 0, 0), axis=-1)
        b = create_revolution(crv, (0, 0, 0), axis=2)
        assert_allclose(np.asarray(a.control_points), np.asarray(b.control_points), rtol=0, atol=0)

    def test_revolution_axis_out_of_range_is_an_index_error(self) -> None:
        with pytest.raises(IndexError, match=r"index 3 is out of bounds for axis 0 with size 3"):
            create_revolution(create_line((1, 0, 0), (1, 0, 1)), (0, 0, 0), axis=3)

    def test_revolution_axis_of_four_coordinates_is_refused(self) -> None:
        # The refusal replaces NumPy's broadcast error: only the type is the old behavior.
        with pytest.raises(ValueError):
            create_revolution(create_line((1, 0, 0), (1, 0, 1)), (0, 0, 0), axis=[0, 0, 1, 0])

    def test_revolution_axis_with_a_nan_revolves_about_z(self) -> None:
        # inherited from main: the normalized axis is NaN, `norm(n) > 1e-14` and `v[2] < 0` are
        # both false, so only the translation is applied and the revolution is about z.
        crv = create_line((1, 0, 0), (1, 0, 1))
        a = create_revolution(crv, (0, 0, 0), axis=[0, float("nan"), 1])
        b = create_revolution(crv, (0, 0, 0), axis=2)
        assert_allclose(np.asarray(a.control_points), np.asarray(b.control_points), rtol=0, atol=0)

    def test_revolution_of_a_rank_two_curve_is_refused(self) -> None:
        crv = _curve([[1, 0], [1, 1]])
        with pytest.raises(ValueError, match=r"create_revolution requires rank == 3, got 2\."):
            create_revolution(crv, (0, 0, 0))

    def test_make_compat_of_one_curve_checks_nothing(self) -> None:
        # inherited from main: fewer than two are returned before any check, a bad axis included.
        crv = create_line()
        (same,) = make_compat(crv, axes=[5])
        assert_allclose(np.asarray(same.control_points), np.asarray(crv.control_points))

    def test_make_compat_of_a_curve_and_a_surface_is_refused(self) -> None:
        with pytest.raises(
            ValueError,
            match=re.escape("All B-splines must have the same parametric dimension, got {1, 2}."),
        ):
            make_compat(create_line(), create_bilinear())

    def test_join_axis_minus_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match=re.escape("axis must be in [0, 1), got -1.")):
            join(create_line(), create_line((1, 0, 0), (2, 0, 0)), -1)

    def test_coons_surface_of_rank_two_curves_is_rank_two(self) -> None:
        s = _random_surface(5)
        flat = Bspline(s.space, np.asarray(s.control_points)[..., :2])
        patch = create_coons_surface(_coons_curves(flat))
        assert patch.rank == 2


class TestRefusalsThatReplaceNumpyErrors:
    """The library refuses where NumPy raised a broadcast error: only the type is pinned."""

    def test_ruled_of_a_rank_one_and_a_rank_three_curve(self) -> None:
        with pytest.raises(ValueError):
            create_ruled(_curve([[0], [1]]), _curve([[0, 0, 0], [1, 0, 0]]))

    def test_ruled_of_a_rank_two_and_a_rank_three_curve(self) -> None:
        with pytest.raises(ValueError):
            create_ruled(_curve([[0, 0], [1, 0]]), _curve([[0, 0, 0], [1, 0, 0]]))

    def test_ruled_of_a_rank_three_and_a_rank_one_curve_is_accepted(self) -> None:
        # inherited from main: the width-1 second curve is broadcast over the three columns.
        ruled = create_ruled(_curve([[0, 0, 0], [1, 0, 0]]), _curve([[5], [7]]))
        assert ruled.dim == 2
        cp = np.asarray(ruled.control_points)
        assert_allclose(cp[:, 1, :], [[5, 5, 5], [7, 7, 7]])
        assert_allclose(cp[:, 0, :], [[0, 0, 0], [1, 0, 0]])

    def test_ruled_of_a_wide_second_curve_is_refused(self) -> None:
        # the reverse of the accepted broadcast is a mismatch of widths.
        with pytest.raises(ValueError):
            create_ruled(_curve([[0], [1]]), _curve([[0, 0], [1, 0]]))

    def test_join_of_a_rank_two_and_a_rank_three_curve(self) -> None:
        with pytest.raises(ValueError):
            join(_curve([[0, 0], [1, 0]]), _curve([[1, 0, 0], [2, 0, 0]]), 0)

    def test_coons_surface_of_curves_of_two_ranks(self) -> None:
        s = _random_surface(6)
        (v0, v1), (u0, u1) = _coons_curves(s)
        flat = Bspline(v0.space, np.asarray(v0.control_points)[..., :2])
        with pytest.raises(ValueError):
            create_coons_surface(((flat, v1), (u0, u1)))

    def test_coons_volume_of_faces_of_two_ranks(self) -> None:
        faces = _faces(_volume())
        flat = Bspline(faces[0][0].space, np.asarray(faces[0][0].control_points)[..., :2])
        with pytest.raises(ValueError):
            create_coons_volume(((flat, faces[0][1]), faces[1], faces[2]))

    def test_knots_one_ulp_apart_leave_unequal_counts_and_ruled_refuses(self) -> None:
        # inherited from main, not endorsed: make_compat matches each merged knot only against
        # the next knot not below it, so `0.1 * 3` and `0.3` (one ulp apart) stay two knots and
        # the results have unequal knot vectors, which create_ruled then refuses with its own
        # message.
        a = _curve([[0, 0, 0], [1, 0, 0], [2, 1, 0]], knots=[0, 0, 0.1 * 3, 1, 1])
        b = _curve([[0, 1, 0], [1, 1, 0], [2, 2, 0]], knots=[0, 0, 0.3, 1, 1])
        ca, cb = make_compat(a, b)
        assert ca.space.num_basis != cb.space.num_basis
        with pytest.raises(ValueError, match="equal numbers of control points"):
            create_ruled(a, b)


class TestMixedPrecision:
    """The dtype of the result of each function given float32 and float64 inputs."""

    @staticmethod
    def _pair() -> tuple[Bspline, Bspline]:
        a = _curve([[0, 0, 0], [1, 0, 0], [2, 1, 0]], degree=2, dtype=np.float64)
        b = _curve([[0, 1, 0], [1, 1, 0]], degree=1, dtype=np.float64)
        return a, b

    def test_make_compat_keeps_each_dtype(self) -> None:
        a, b = self._pair()
        c, d = make_compat(_f32(a), b)
        assert (c.dtype, d.dtype) == (np.float32, np.float64)
        c, d = make_compat(a, _f32(b))
        assert (c.dtype, d.dtype) == (np.float64, np.float32)

    def test_ruled_takes_the_first_dtype(self) -> None:
        a, b = self._pair()
        assert create_ruled(_f32(a), b).dtype == np.float32
        assert create_ruled(a, _f32(b)).dtype == np.float64

    def test_ruled_refuses_a_float64_value_beyond_float32(self) -> None:
        # new in the C++ port: the conversion to the first dtype is checked, where the Python
        # implementation stored infinity and warned (RuntimeWarning: overflow encountered in cast).
        a, _ = self._pair()
        huge = _curve(np.full((2, 3), 1e300), dtype=np.float64)
        with pytest.raises(ValueError, match="outside the range of the target floating-point type"):
            create_ruled(_f32(a), huge)

    def test_join_of_curves_is_float64_in_either_order(self) -> None:
        left, right = _curve([[0, 0, 0], [1, 0, 0]]), _curve([[1, 0, 0], [2, 1, 0]])
        assert join(_f32(left), right, 0).dtype == np.float64
        assert join(left, _f32(right), 0).dtype == np.float64

    def test_join_of_surfaces_float32_then_float64_is_refused(self) -> None:
        # inherited from main: the second axis of the first stays float32 while the join axis is
        # float64, which a space refuses, after every other check has passed.
        with pytest.raises(ValueError, match=re.escape(_DTYPE_MESSAGE)):
            join(_surface(np.float32), _surface(x0=1.0), 0)

    def test_join_of_surfaces_float64_then_float32_is_float64(self) -> None:
        assert join(_surface(), _surface(np.float32, x0=1.0), 0).dtype == np.float64

    def test_join_of_surfaces_of_one_dtype_stays_in_it(self) -> None:
        assert join(_surface(np.float32), _surface(np.float32, x0=1.0), 0).dtype == np.float32

    def test_join_float32_then_float64_still_checks_the_seam_first(self) -> None:
        # inherited from main: the dtype refusal comes last, so a seam that does not meet is the
        # error a caller sees.
        with pytest.raises(ValueError, match="Shared boundary mismatch"):
            join(_surface(np.float32), _surface(x0=1.0, shift=0.5), 0)

    def test_join_float32_then_float64_still_checks_the_axis_first(self) -> None:
        with pytest.raises(ValueError, match=re.escape("axis must be in [0, 2), got 5.")):
            join(_surface(np.float32), _surface(x0=1.0), 5)

    def test_sweep_of_mixed_dtypes_is_refused(self) -> None:
        # inherited from main (create_sweep stays in Python and keeps its behavior by itself).
        with pytest.raises(ValueError, match=re.escape(_DTYPE_MESSAGE)):
            create_sweep(_curve([[0, 0, 0], [1, 0, 0]], dtype=np.float32), create_line())

    def test_revolution_of_a_float32_curve_is_refused(self) -> None:
        # inherited from main: the arc is float64 and the result's spaces would be of two dtypes.
        crv = _curve([[1, 0, 0], [1, 0, 1]], dtype=np.float32)
        with pytest.raises(ValueError, match=re.escape(_DTYPE_MESSAGE)):
            create_revolution(crv, (0, 0, 0))

    def test_revolution_of_a_float32_curve_reports_the_earlier_refusals_first(self) -> None:
        # inherited from main: the dtype is refused last, so the rank is the error a caller sees.
        crv = _curve([[1, 0], [1, 1]], dtype=np.float32)
        with pytest.raises(ValueError, match="requires rank == 3"):
            create_revolution(crv, (0, 0, 0))

    def test_coons_surface_takes_the_dtype_of_the_first_curve(self) -> None:
        s = _random_surface(7)
        (v0, v1), (u0, u1) = _coons_curves(s)
        assert create_coons_surface(((_f32(v0), v1), (u0, u1))).dtype == np.float32
        assert create_coons_surface(((v0, _f32(v1)), (u0, u1))).dtype == np.float64

    def test_coons_volume_takes_the_dtype_of_the_first_face(self) -> None:
        faces = _faces(_volume())
        assert create_coons_volume(
            ((faces[0][0], _f32(faces[0][1])), faces[1], faces[2])
        ).dtype == (np.float64)
        assert create_coons_volume(
            ((_f32(faces[0][0]), faces[0][1]), faces[1], faces[2])
        ).dtype == (np.float32)

    def test_coons_volume_of_a_float32_second_face_grades_at_float32(self) -> None:
        # the float32 face's corners are the float64 ones rounded: a float64 grade would refuse
        # them, the coarsest dtype among the inputs accepts them.
        faces = _faces(_volume())
        volume = create_coons_volume(((faces[0][0], _f32(faces[0][1])), faces[1], faces[2]))
        assert volume.dim == 3

    def test_extrusion_of_float32_stays_float32(self) -> None:
        # create_extrusion stays in Python.
        assert create_extrusion(
            _curve([[0, 0, 0], [1, 0, 0]], dtype=np.float32), [0, 0, 1]
        ).dtype == (np.float32)


class TestHandles:
    """Results built by the library are ordinary B-splines."""

    def test_results_carry_the_spaces_of_their_handle(self) -> None:
        ruled = create_ruled(create_circle(0.5), create_circle(1.0))
        assert [s.degree for s in ruled.space.spaces] == [2, 1]
        assert ruled.space.num_basis == (9, 2)
        assert ruled.is_rational

    def test_a_result_is_a_working_bspline(self) -> None:
        # the corner (1, 1, 1) is the control point itself: exact, no tolerance.
        corner = create_trilinear().evaluate(np.array([[1.0, 1.0, 1.0]]))
        assert_allclose(corner, [0.5, 0.5, 0.5], rtol=0, atol=0)


def test_make_compat_returns_an_unchanged_input_as_the_same_object() -> None:
    """An input the compatibility leaves as it is comes back as that object, as before the port.

    Two lines are already compatible, so both come back as they were given; a line against a
    circle is elevated, so the line is a new object and the circle is the one given. A single
    B-spline is returned untouched.
    """
    line, other, circle = create_line(), create_line([0, 1, 0], [1, 1, 0]), create_circle()
    assert make_compat(line)[0] is line
    same = make_compat(line, other)
    assert same[0] is line
    assert same[1] is other
    elevated = make_compat(line, circle)
    assert elevated[0] is not line
    assert elevated[1] is circle


def test_mixed_make_compat_returns_unchanged_inputs_as_the_same_objects() -> None:
    """The float32 input of a mixed pair that needs no change is the object given, as before."""
    a = create_line()
    b = _f32(create_line([0, 1, 0], [1, 1, 0]))
    out = make_compat(a, b)
    assert out[0] is a
    assert out[1] is b


def test_mixed_make_compat_refuses_knots_that_round_to_one_float32() -> None:
    """A double knot at ``0.3`` and its float32 rounding are two knots in float64.

    Each side receives the other's two copies, and on the way back to float32 the four round to
    one value of multiplicity 4 at degree 1. The previous release inserted in float32 and refused
    with this message; the port refuses at the same call rather than return a space no later
    operation accepts.
    """
    cp = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]
    knots = [0.0, 0.0, 0.3, 0.3, 1.0, 1.0]
    a = _curve(cp, knots=knots)
    b = _curve(cp, knots=knots, dtype=np.float32)
    with pytest.raises(ValueError, match="exceed the maximum multiplicity of 2"):
        make_compat(a, b)


def test_a_mixed_coons_patch_with_a_knot_rounded_to_float32_is_built() -> None:
    """New in the port: a float32 reading of an interior knot no longer breaks the blend.

    The two curves free in u carry an interior knot at ``0.3``, ``C_u0`` in float32 and ``C_u1``
    in float64. The previous release made the pair compatible in their own precisions, so the
    two readings of the knot counted as two on one side only, and the ruled term failed with a
    NumPy broadcast error; the port converts every curve to the first curve's dtype first, so
    both carry one knot and the patch interpolates its sides.
    """
    t = np.array([0.0, 0.15, 0.65, 1.0])
    bottom = np.stack([t, np.zeros(4), np.zeros(4)], axis=1)
    top = np.stack([t, np.ones(4), 0.2 * np.sin(np.pi * t)], axis=1)
    knots = [0.0, 0.0, 0.0, 0.3, 1.0, 1.0, 1.0]
    c_u0 = _curve(bottom, degree=2, knots=knots, dtype=np.float32)
    c_u1 = _curve(top, degree=2, knots=knots)
    c_v0 = create_line([0, 0, 0], [0, 1, 0])
    c_v1 = create_line([1, 0, 0], [1, 1, float(top[-1, 2])])
    patch = create_coons_surface(((c_v0, c_v1), (c_u0, c_u1)))
    assert patch.dtype == np.float64
    u = np.linspace(0.0, 1.0, 7)
    on_top = patch.evaluate(np.stack([u, np.ones(7)], axis=1))
    assert_allclose(on_top, c_u1.evaluate(u), atol=4 * float(np.finfo(np.float32).eps))


def test_join_takes_a_bool_axis_as_the_integer_it_equals() -> None:
    """Changed in the port: ``False`` is axis 0 (the previous release raised ``TypeError``)."""
    c1, c2 = create_line([0, 0, 0], [1, 0, 0]), create_line([1, 0, 0], [2, 0, 0])
    assert join(c1, c2, axis=False).dim == 1
    with pytest.raises(ValueError, match=re.escape("axis must be in [0, 1), got 1.")):
        join(c1, c2, axis=True)
