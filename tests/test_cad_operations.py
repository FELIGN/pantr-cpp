"""Tests for CAD operations: create_extrusion and create_ruled."""

from __future__ import annotations

import re

import numpy as np
import numpy.typing as npt
import pytest
from numpy.testing import assert_allclose

from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D
from pantr.cad import create_bilinear, create_circle, create_extrusion, create_line, create_ruled

_RANK_3D = 3


class TestExtrude:
    """Test the extrude operation."""

    def test_line_to_surface(self) -> None:
        """Test extruding a line into a surface."""
        crv = create_line([0, 0, 0], [1, 0, 0])
        srf = create_extrusion(crv, [0, 0, 1])
        assert srf.dim == 2
        assert srf.rank == _RANK_3D
        assert not srf.is_rational

    def test_extrude_degree(self) -> None:
        """Test that the new direction has degree 1."""
        crv = create_line([0, 0, 0], [1, 0, 0])
        srf = create_extrusion(crv, [0, 1, 0])
        assert srf.degree == (1, 1)

    def test_extrude_evaluate_corners(self) -> None:
        """Test evaluation at the four corners of an extruded line."""
        crv = create_line([0, 0, 0], [2, 0, 0])
        srf = create_extrusion(crv, [0, 3, 0])
        pts = srf.evaluate(
            np.array(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                ]
            )
        )
        assert_allclose(pts[0], [0, 0, 0], atol=1e-14)
        assert_allclose(pts[1], [2, 0, 0], atol=1e-14)
        assert_allclose(pts[2], [0, 3, 0], atol=1e-14)
        assert_allclose(pts[3], [2, 3, 0], atol=1e-14)

    def test_extrude_circle_to_cylinder(self) -> None:
        """Test extruding a circle produces a cylinder."""
        crv = create_circle(radius=2.0)
        cyl = create_extrusion(crv, [0, 0, 5])
        assert cyl.dim == 2
        assert cyl.is_rational
        # Evaluate on a grid and check all points lie on the cylinder
        u = np.linspace(0, 1, 20)
        v = np.array([0.0, 0.5, 1.0])
        params = np.array([[ui, vi] for ui in u for vi in v])
        pts = cyl.evaluate(params)
        radii = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
        assert_allclose(radii, 2.0, atol=1e-13)
        # z should range from 0 to 5
        assert_allclose(pts[params[:, 1] == 0.0, 2], 0.0, atol=1e-14)
        assert_allclose(pts[params[:, 1] == 1.0, 2], 5.0, atol=1e-13)

    def test_extrude_surface_to_volume(self) -> None:
        """Test extruding a surface into a volume."""
        srf = create_bilinear()
        vol = create_extrusion(srf, [0, 0, 1])
        assert vol.dim == _RANK_3D
        assert vol.degree == (1, 1, 1)

    def test_extrude_volume_raises(self) -> None:
        """Test that extruding a volume raises ValueError."""
        vol = create_extrusion(create_bilinear(), [0, 0, 1])
        with pytest.raises(ValueError, match="dim"):
            create_extrusion(vol, [0, 0, 1])

    def test_extrude_2d_displacement(self) -> None:
        """Test that 2D displacement is zero-padded to 3D."""
        crv = create_line([0, 0, 0], [1, 0, 0])
        srf = create_extrusion(crv, [0, 1])
        pt = srf.evaluate(np.array([[0.5, 1.0]]))
        assert_allclose(pt, [0.5, 1.0, 0.0], atol=1e-14)


class TestRuled:
    """Test the ruled surface/volume construction."""

    def test_ruled_two_lines(self) -> None:
        """Test ruled surface between two parallel lines."""
        c1 = create_line([0, 0, 0], [1, 0, 0])
        c2 = create_line([0, 1, 0], [1, 1, 0])
        srf = create_ruled(c1, c2)
        assert srf.dim == 2
        assert srf.degree == (1, 1)
        assert not srf.is_rational

    def test_ruled_evaluate_corners(self) -> None:
        """Test evaluation at the four parametric corners."""
        c1 = create_line([0, 0, 0], [3, 0, 0])
        c2 = create_line([0, 2, 0], [3, 2, 0])
        srf = create_ruled(c1, c2)
        pts = srf.evaluate(
            np.array(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                ]
            )
        )
        assert_allclose(pts[0], [0, 0, 0], atol=1e-14)
        assert_allclose(pts[1], [3, 0, 0], atol=1e-14)
        assert_allclose(pts[2], [0, 2, 0], atol=1e-14)
        assert_allclose(pts[3], [3, 2, 0], atol=1e-14)

    def test_ruled_annulus(self) -> None:
        """Test ruled surface between two circles forms an annulus."""
        inner = create_circle(radius=1.0)
        outer = create_circle(radius=2.0)
        annulus = create_ruled(inner, outer)
        assert annulus.dim == 2
        assert annulus.is_rational
        # Evaluate and check radii
        u = np.linspace(0, 1, 30)
        params_inner = np.column_stack([u, np.zeros_like(u)])
        params_outer = np.column_stack([u, np.ones_like(u)])
        pts_inner = annulus.evaluate(params_inner)
        pts_outer = annulus.evaluate(params_outer)
        radii_inner = np.sqrt(pts_inner[:, 0] ** 2 + pts_inner[:, 1] ** 2)
        radii_outer = np.sqrt(pts_outer[:, 0] ** 2 + pts_outer[:, 1] ** 2)
        assert_allclose(radii_inner, 1.0, atol=1e-13)
        assert_allclose(radii_outer, 2.0, atol=1e-13)

    def test_ruled_different_degrees(self) -> None:
        """Test that ruled makes inputs compatible (different degrees)."""
        c1 = create_line([0, 0, 0], [1, 0, 0])  # degree 1
        c2 = create_circle(angle=np.pi / 2)  # degree 2
        srf = create_ruled(c1, c2)
        # Both should have been elevated to degree 2
        assert srf.degree[0] == 2

    def test_ruled_mixed_rational(self) -> None:
        """Test that mixing rational and non-rational promotes correctly."""
        c1 = create_line([1, 0, 0], [2, 0, 0])  # non-rational
        c2 = create_circle(radius=1.5, angle=np.pi / 4)  # rational
        srf = create_ruled(c1, c2)
        assert srf.is_rational

    def test_ruled_different_dim_raises(self) -> None:
        """Test that different dimensions raises ValueError."""
        crv = create_line([0, 0, 0], [1, 0, 0])
        srf = create_bilinear()
        with pytest.raises(ValueError, match="same dim"):
            create_ruled(crv, srf)

    def test_ruled_volume_from_surfaces(self) -> None:
        """Test ruled volume between two surfaces."""
        corners1 = np.array([[[0, 0, 0], [0, 1, 0]], [[1, 0, 0], [1, 1, 0]]], dtype=np.float64)
        corners2 = np.array([[[0, 0, 1], [0, 1, 1]], [[1, 0, 1], [1, 1, 1]]], dtype=np.float64)
        s1 = create_bilinear(corners1)
        s2 = create_bilinear(corners2)
        vol = create_ruled(s1, s2)
        assert vol.dim == _RANK_3D
        assert vol.degree == (1, 1, 1)
        # Center should be at (0.5, 0.5, 0.5)
        pt = vol.evaluate(np.array([[0.5, 0.5, 0.5]]))
        assert_allclose(pt, [0.5, 0.5, 0.5], atol=1e-14)

    @staticmethod
    def _bilinear_surface(
        u_knots: list[float], v_knots: list[float], cp: npt.NDArray[np.float64]
    ) -> Bspline:
        """Build a degree-(1, 1) surface on the given knots.

        Args:
            u_knots (list[float]): Knots in the first direction.
            v_knots (list[float]): Knots in the second direction.
            cp (npt.NDArray[np.float64]): Control net of shape ``(nu, nv, 3)``.

        Returns:
            Bspline: The surface.
        """
        space = BsplineSpace([BsplineSpace1D(u_knots, 1), BsplineSpace1D(v_knots, 1)])
        return Bspline(space, cp)

    @staticmethod
    def _net(nu: int, nv: int) -> npt.NDArray[np.float64]:
        """Build a control net whose points are all distinct.

        Args:
            nu (int): Points in the first direction.
            nv (int): Points in the second direction.

        Returns:
            npt.NDArray[np.float64]: Shape ``(nu, nv, 3)``.
        """
        i, j = np.meshgrid(np.arange(nu), np.arange(nv), indexing="ij")
        return np.stack([i, j, (i + 1.0) * (j + 2.0)], axis=-1).astype(np.float64)

    @pytest.mark.parametrize(
        ("u_knots", "nu", "shapes"),
        [
            ([0.0, 0.0, 0.0, 1.0, 1.0], 3, r"\(3, 2\) and \(2, 3\)"),  # a zero-support end function
            ([-1.0, 0.0, 0.0, 1.0, 1.0, 2.0], 4, r"\(4, 2\) and \(2, 4\)"),  # knots off the domain
        ],
        ids=["zero-support-end-function", "knots-outside-the-domain"],
    )
    def test_ruled_refuses_nets_of_one_size_and_different_shapes(
        self, u_knots: list[float], nu: int, shapes: str
    ) -> None:
        """Test that transposed nets, unmatched by make_compat, are refused, not paired by index.

        `make_compat` matches interior knot classes only, so the second input (the first one
        transposed) keeps a net of the transposed shape, with the same number of points.
        """
        cp = self._net(nu, 2)
        s1 = self._bilinear_surface(u_knots, [0.0, 0.0, 1.0, 1.0], cp)
        s2 = self._bilinear_surface([0.0, 0.0, 1.0, 1.0], u_knots, np.swapaxes(cp, 0, 1))
        with pytest.raises(ValueError, match=rf"control nets have the same shape.*got {shapes}\."):
            create_ruled(s1, s2)

    def test_ruled_nets_of_one_shape_keep_both_inputs_as_end_faces(self) -> None:
        """Test that the shape check leaves equal shapes alone: the end faces are the inputs.

        The oracle is the inputs evaluated directly. The bound is the one the C++ suite derives
        for a ruled B-spline of two degree-1 inputs, ``4 (3 d + 2^d) eps max |cp|`` with ``d = 2``
        (``cpp/tests/test_cad_operations.cpp``, its header), taken over the larger net, whose
        coordinates reach ``max |cp| + 5``. The result is a volume, but at ``t = 0`` or ``1`` its
        degree-1 basis in ``t`` is exactly ``(1, 0)`` or ``(0, 1)``, which adds no rounding, so
        two inexact directions remain and ``d = 2`` holds.
        """
        cp = self._net(3, 2)
        u_knots = [0.0, 0.0, 0.0, 1.0, 1.0]
        s1 = self._bilinear_surface(u_knots, [0.0, 0.0, 1.0, 1.0], cp)
        s2 = self._bilinear_surface(u_knots, [0.0, 0.0, 1.0, 1.0], cp + np.array([0.0, 0.0, 5.0]))
        r = create_ruled(s1, s2)
        uu, vv = np.meshgrid(np.linspace(0, 1, 7), np.linspace(0, 1, 7), indexing="ij")
        uv = np.column_stack([uu.ravel(), vv.ravel()])
        d = 2
        atol = 4 * (3 * d + 2**d) * np.finfo(np.float64).eps * (np.abs(cp).max() + 5.0)
        for t, src in ((0.0, s1), (1.0, s2)):
            face = r.evaluate(np.column_stack([uv, np.full(len(uv), t)]))
            assert_allclose(face, src.evaluate(uv), rtol=0, atol=atol)

    _END_CLASS_CP = np.array([[0.0, 0.0], [1.0, 2.0], [2.0, 0.0]])
    """The control points of issue #54's curves, small integers so that every array is exact."""
    _END_CLASS_SHIFT = np.array([0.0, 3.0])
    """The offset of the second curve's control points."""

    @staticmethod
    def _degree1_curve(knots: list[float], cp: npt.NDArray[np.float64]) -> Bspline:
        """Build a degree-1 curve on the given knots.

        Args:
            knots (list[float]): The knot vector.
            cp (npt.NDArray[np.float64]): Control points of shape ``(n, rank)``.

        Returns:
            Bspline: The curve.
        """
        return Bspline(BsplineSpace([BsplineSpace1D(np.array(knots, dtype=np.float64), 1)]), cp)

    @pytest.mark.parametrize(
        ("knots1", "classes1", "knots2", "classes2"),
        [
            # Issue #54's case: a first class of three knots against a knot past the domain's end.
            ([0.0, 0.0, 0.0, 1.0, 1.0], "(3, 2)", [0.0, 0.0, 1.0, 1.0, 2.0], "(2, 3)"),
            # No end knot repeated more than `degree + 1` times: a knot before the domain's start.
            ([0.0, 0.0, 1.0, 1.0, 2.0], "(2, 3)", [-1.0, 0.0, 0.0, 1.0, 1.0], "(3, 2)"),
        ],
        ids=["long-start-vs-past-end", "past-end-vs-before-start"],
    )
    @pytest.mark.parametrize("swap", [False, True], ids=["in-order", "swapped"])
    def test_ruled_refuses_end_knot_classes_that_differ(
        self, knots1: list[float], classes1: str, knots2: list[float], classes2: str, swap: bool
    ) -> None:
        """Test that nets of one shape on different end knot classes are refused (issue #54).

        `make_compat` leaves end classes as given, and the result used to take the first input's
        knots, so its ``t = 1`` edge was not the second input. A class counts the knots at or
        before the domain's start, and at or after its end.
        """
        a = self._degree1_curve(knots1, self._END_CLASS_CP)
        b = self._degree1_curve(knots2, self._END_CLASS_CP + self._END_CLASS_SHIFT)
        first, second = (classes2, classes1) if swap else (classes1, classes2)
        message = f"same end knot classes after making them compatible, got {first} and {second}"
        with pytest.raises(ValueError, match=re.escape(f"{message} in direction 0")):
            create_ruled(*((b, a) if swap else (a, b)))

    @pytest.mark.parametrize(
        "knots",
        [[0.0, 0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0, 2.0]],
        ids=["long-start", "past-end"],
    )
    def test_ruled_equal_end_knot_classes_stack_the_nets(self, knots: list[float]) -> None:
        """Test that equal end classes give the stacked nets on the first input's knots, exactly.

        The controls of issue #54: the data are small integers, so the arrays are exact, and the
        result is the one returned before the end classes were checked.
        """
        a = self._degree1_curve(knots, self._END_CLASS_CP)
        b = self._degree1_curve(knots, self._END_CLASS_CP + self._END_CLASS_SHIFT)
        r = create_ruled(a, b)
        np.testing.assert_array_equal(r.space.spaces[0].knots, knots)
        np.testing.assert_array_equal(r.space.spaces[1].knots, [0.0, 0.0, 1.0, 1.0])
        np.testing.assert_array_equal(
            r.control_points, np.stack([a.control_points, b.control_points], axis=1)
        )

    @pytest.mark.parametrize("swap", [False, True], ids=["in-order", "swapped"])
    def test_ruled_accepts_end_knots_differing_within_one_class(self, swap: bool) -> None:
        """Test that end knots differing in value within one class are accepted (issue #54).

        A third knot a quarter of the space tolerance above the start (unsnapped) is in the first
        class, so ``[0, 0, d, 1, 1]`` counts ``(3, 2)`` as ``[0, 0, 0, 1, 1]`` does, though a
        count of knots exactly at or before 0 would give 2. The result is the two nets stacked
        on the first input's knots.
        """
        zero = BsplineSpace1D(np.array([0.0, 0.0, 0.0, 1.0, 1.0]), 1)
        d = 0.25 * zero.tolerance
        dust = BsplineSpace1D(np.array([0.0, 0.0, d, 1.0, 1.0]), 1, snap_knots=False)
        assert d > 0.0
        assert dust.knots[2] == d
        cp_up = self._END_CLASS_CP + self._END_CLASS_SHIFT
        a = Bspline(BsplineSpace([zero]), self._END_CLASS_CP)
        b = Bspline(BsplineSpace([dust]), cp_up)
        first, second = (b, a) if swap else (a, b)
        r = create_ruled(first, second)
        np.testing.assert_array_equal(r.space.spaces[0].knots, first.space.spaces[0].knots)
        nets = [cp_up, self._END_CLASS_CP] if swap else [self._END_CLASS_CP, cp_up]
        np.testing.assert_array_equal(r.control_points, np.stack(nets, axis=1))
