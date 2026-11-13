"""Tests for the B-spline fitting functions against oracles that do not share their route.

Each public fitting function has at least one test whose expected value is computed another way
than the code computes it; the test's docstring names the oracle. The routes used are:

- **A polynomial of the space, reproduced exactly.** A polynomial of degree at most the degree of
  the space lies in the space, so interpolating, fitting, or quasi-interpolating samples of it must
  give back a spline that equals it. The spline the library returns is evaluated in exact
  rational arithmetic (the Cox-de Boor recursion of :mod:`tests.oracles.bspline_exact`, not the
  triangle the library runs) at points that are not nodes, and compared with the polynomial
  evaluated exactly.
- **Exact rational linear algebra.** The collocation matrix is built in :class:`~fractions.Fraction`
  from the same exact basis, and the coefficients are the exact solution (the normal equations for
  a rectangular system), so the library's banded-QR/LU/SVD solves are compared with an exact
  answer, not a second floating-point one.
- **Exact rational Gram matrices.** The Gram matrix ``G_ij = integral N_i N_j`` and the load
  ``b_i = integral N_i f`` are integrated span by span with an interpolatory quadrature on
  interior equispaced nodes whose weights solve the moment equations in rationals, so nothing of
  the library's Gauss rules is shared. The L2 projection of a polynomial that is *not* in the space
  is ``G^-1 b``; boundary interpolation replaces the first and the last row by the basis at the
  ends, and for a separable function on a tensor space the projection is the outer product of the
  one-dimensional ones.
- **An invariant.** ``Q B = B`` for the quasi-interpolant.

Tolerances. ``u`` is the unit roundoff of the space's dtype. Every solve is backward stable, so the
forward error of the coefficients is at most ``8 (m + n + 2) u kappa ||c||_2`` for a system of
``m`` equations in ``n`` unknowns with condition number ``kappa`` (the constant 8 and the form are
those of ``cpp/tests/test_bezier_fit.cpp``: they absorb the rounding of the data and the
``6 p + 1`` roundings of each matrix entry for the degrees used here). ``kappa_2(A)`` is bounded
from the exact inverse of the normal matrix ``N = A^T A`` by
``sqrt(n ||N||_inf ||N^-1||_inf)``; for a Kronecker product ``kappa`` is the product of the
factors' and the size term the sum of theirs, and for an L2 projection the size term adds the number
of quadrature points (each mass-matrix entry sums that many positive terms). A spline of the space
evaluated at a point is a convex combination of its coefficients, so the same bound holds for its
value there. The quasi-interpolant adds the rounding of its local blocks (below).
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from fractions import Fraction
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from pantr.bspline import (
    Bspline,
    BsplineSpace,
    BsplineSpace1D,
    create_uniform_space,
    fit_bspline,
    get_greville_abscissae,
    interpolate_bspline,
    l2_project_bspline,
    quasi_interpolate_bspline,
)
from pantr.quad import PointsLattice
from tests.oracles import bspline_exact

Exact = list[list[Fraction]]
"""A matrix of exact rationals, row-major."""

K_SLACK = 8.0
"""The constant of the header: slack over the backward error of one solve."""


# ---------------------------------------------------------------------------
# Exact linear algebra
# ---------------------------------------------------------------------------


def _transpose(a: Exact) -> Exact:
    return [list(col) for col in zip(*a, strict=True)]


def _matmul(a: Exact, b: Exact) -> Exact:
    bt = _transpose(b)
    return [
        [sum((x * y for x, y in zip(row, col, strict=True)), Fraction(0)) for col in bt]
        for row in a
    ]


def _solve_exact(a: Exact, b: Exact) -> Exact:
    """Solve the square system ``a x = b`` by Gauss-Jordan elimination on exact entries.

    Args:
        a (Exact): ``(n, n)``, non-singular.
        b (Exact): ``(n, k)``.

    Returns:
        Exact: The ``(n, k)`` solution.
    """
    n = len(a)
    a = [list(r) for r in a]
    b = [list(r) for r in b]
    for col in range(n):
        piv = next(r for r in range(col, n) if a[r][col] != 0)  # StopIteration: singular
        a[col], a[piv] = a[piv], a[col]
        b[col], b[piv] = b[piv], b[col]
        d = a[col][col]
        a[col] = [v / d for v in a[col]]
        b[col] = [v / d for v in b[col]]
        for r in range(n):
            if r != col and a[r][col] != 0:
                f = a[r][col]
                a[r] = [x - f * y for x, y in zip(a[r], a[col], strict=True)]
                b[r] = [x - f * y for x, y in zip(b[r], b[col], strict=True)]
    return b


def _lstsq_exact(a: Exact, b: Exact) -> Exact:
    """Return the least-squares solution of a full-column-rank system (the normal equations)."""
    at = _transpose(a)
    return _solve_exact(_matmul(at, a), _matmul(at, b))


def _norm_inf(a: Exact) -> float:
    return float(max(sum((abs(v) for v in row), Fraction(0)) for row in a))


def _frobenius(a: Exact) -> float:
    return math.sqrt(float(sum((v * v for row in a for v in row), Fraction(0))))


def _kappa_bound(a: Exact) -> float:
    """Return an upper bound of ``kappa_2`` of a full-column-rank matrix (module docstring)."""
    n = len(a[0])
    normal = _matmul(_transpose(a), a)
    inverse = _solve_exact(normal, [[Fraction(int(i == j)) for j in range(n)] for i in range(n)])
    return math.sqrt(n * _norm_inf(normal) * _norm_inf(inverse))


def _solve_bound(size_term: int, kappa: float, x_norm: float, dtype: npt.DTypeLike) -> float:
    return K_SLACK * size_term * float(np.finfo(dtype).eps) / 2.0 * kappa * x_norm


def _exact_collocation(space: BsplineSpace1D, nodes: npt.NDArray[Any]) -> Exact:
    """Return the collocation matrix ``N_j(x_i)`` from the exact basis (a periodic space wraps)."""
    knots = np.asarray(space.knots)
    n = int(space.num_basis)
    rows: Exact = []
    for x in nodes:
        basis = bspline_exact.ExactBasis(knots, space.degree, x)
        first = basis.span - space.degree
        row = [Fraction(0)] * n
        for a, v in enumerate(basis.local()):
            row[(first + a) % n] += v
        rows.append(row)
    return rows


def _exact_value(spline: Bspline, x: Sequence[float]) -> list[Fraction]:
    """Return the exact value (all coordinates) of a spline, from its stored control points."""
    spaces = spline.space.spaces
    ctrl = np.asarray(spline.control_points)
    return bspline_exact.homogeneous(
        [s.knots for s in spaces], [s.degree for s in spaces], ctrl, list(x), [0] * len(spaces)
    )


def _poly_exact(coeffs: Sequence[Fraction], x: Fraction) -> Fraction:
    return sum((c * x**k for k, c in enumerate(coeffs)), Fraction(0))


def _poly_float(coeffs: Sequence[Fraction], x: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    """Evaluate a polynomial in float64 (Horner) and round to the dtype of ``x``."""
    result = np.zeros(x.shape, dtype=np.float64)
    for c in reversed(coeffs):
        result = result * x.astype(np.float64) + float(c)
    return result.astype(x.dtype)


def _dyadic_poly(degree: int, salt: int = 0) -> list[Fraction]:
    """Return the coefficients (-1)^k (k + 2 + salt) / 2^k, k = 0 .. degree."""
    return [Fraction((-1) ** k * (k + 2 + salt), 2**k) for k in range(degree + 1)]


def _knots(
    degree: int, interior: Sequence[float], lo: float = 0.0, hi: float = 1.0
) -> npt.NDArray[Any]:
    return np.array([lo] * (degree + 1) + list(interior) + [hi] * (degree + 1), dtype=np.float64)


def _space_from_knots(*pairs: tuple[npt.NDArray[Any], int]) -> BsplineSpace:
    return BsplineSpace([BsplineSpace1D(k, p) for k, p in pairs])


def _probe_points(lo: float, hi: float) -> list[float]:
    """Return points of a domain that are not nodes (nor knots of the meshes used here)."""
    return [lo + (hi - lo) * t for t in (0.0, 0.013, 0.26, 0.4, 0.77, 0.999, 1.0)]


def _assert_within(got: Fraction, want: Fraction, bound: float, what: str) -> None:
    assert abs(got - want) <= Fraction(bound), (
        f"{what}: |{float(got)} - {float(want)}| = {float(abs(got - want))} > {bound}"
    )


_SPACES_1D: dict[str, Callable[[], BsplineSpace]] = {
    "uniform cubic, 4 intervals": lambda: create_uniform_space(3, 4),
    "nonuniform quartic": lambda: _space_from_knots((_knots(4, [0.125, 0.5, 0.75]), 4)),
    "quadratic, C0 at a double knot": lambda: _space_from_knots(
        (_knots(2, [0.25, 0.5, 0.5, 0.75]), 2)
    ),
    "cubic on [2, 5]": lambda: create_uniform_space(3, 4, domain=(2.0, 5.0)),
    "float32 cubic": lambda: create_uniform_space(3, 4, dtype=np.float32),
}


# ---------------------------------------------------------------------------
# interpolate_bspline
# ---------------------------------------------------------------------------


class TestInterpolateOracles:
    """Oracles for :func:`interpolate_bspline`."""

    @pytest.mark.parametrize("name", list(_SPACES_1D))
    def test_polynomial_of_the_space_degree_is_reproduced_exactly(self, name: str) -> None:
        """Oracle: an exactly reproduced polynomial, with the exact rational solution.

        The interpolant of ``f`` (a polynomial of the space's degree) at the Greville abscissae is
        ``f`` itself. The library's spline is evaluated in exact arithmetic at points that are not
        nodes and compared with ``f`` evaluated exactly, and its coefficients with the exact
        solution of the exact collocation system. Bug caught: a transposed or mis-scaled
        collocation matrix, nodes that are not the Greville abscissae, a banded solve that drifts
        from the dense one, a coefficient layout that is not the basis order.
        """
        space = _SPACES_1D[name]()
        s1 = space.spaces[0]
        poly = _dyadic_poly(s1.degree)
        nodes = get_greville_abscissae(s1)
        a = _exact_collocation(s1, nodes)
        c_exact = _solve_exact(a, [[_poly_exact(poly, bspline_exact.exact(x))] for x in nodes])
        n = len(nodes)
        bound = _solve_bound(2 * n + 2, _kappa_bound(a), _frobenius(c_exact), space.dtype)

        spline = interpolate_bspline(
            lambda lat: _poly_float(poly, lat.get_all_points()[:, 0]), space
        )

        ctrl = np.asarray(spline.control_points)
        assert ctrl.dtype == space.dtype
        for i in range(n):
            _assert_within(Fraction(float(ctrl[i, 0])), c_exact[i][0], bound, f"coefficient {i}")
        lo, hi = (float(v) for v in s1.domain)
        for x in _probe_points(lo, hi):
            want = _poly_exact(poly, bspline_exact.exact(x))
            _assert_within(_exact_value(spline, [x])[0], want, bound, f"value at {x}")

    def test_vector_valued_function_reproduces_each_component(self) -> None:
        """Oracle: an exactly reproduced polynomial, one per component (rank 2).

        Bug caught: components interleaved or swapped in the coefficient array, one component
        solved against the other's data.
        """
        space = create_uniform_space(3, 4)
        s1 = space.spaces[0]
        p0, p1 = _dyadic_poly(3), _dyadic_poly(3, salt=5)
        nodes = get_greville_abscissae(s1)
        a = _exact_collocation(s1, nodes)
        c0 = _solve_exact(a, [[_poly_exact(p0, bspline_exact.exact(x))] for x in nodes])
        c1 = _solve_exact(a, [[_poly_exact(p1, bspline_exact.exact(x))] for x in nodes])
        bound = _solve_bound(
            2 * len(nodes) + 2, _kappa_bound(a), _frobenius(c0) + _frobenius(c1), space.dtype
        )

        def func(lat: PointsLattice) -> npt.NDArray[Any]:
            x = lat.get_all_points()[:, 0]
            return np.stack([_poly_float(p0, x), _poly_float(p1, x)], axis=-1)

        spline = interpolate_bspline(func, space)
        assert spline.rank == 2
        ctrl = np.asarray(spline.control_points)
        for i in range(len(nodes)):
            _assert_within(
                Fraction(float(ctrl[i, 0])), c0[i][0], bound, f"component 0, coefficient {i}"
            )
            _assert_within(
                Fraction(float(ctrl[i, 1])), c1[i][0], bound, f"component 1, coefficient {i}"
            )

    def test_bipolynomial_on_a_tensor_space_is_reproduced_exactly(self) -> None:
        """Oracle: an exactly reproduced bi-polynomial, exact Kronecker solve.

        ``f(x, y) = x^3 y^2 - x y + 2 y`` lies in the space of degrees (3, 2); the interpolant at
        the Greville grid equals it. Expected coefficients: the exact solution of each direction's
        collocation system applied in turn. Bug caught: the two directions' systems applied to
        the wrong axes (the grids have different sizes, 7 and 6), a lattice in F order.
        """
        space = _space_from_knots(
            (_knots(3, [0.25, 0.5, 0.75]), 3), (_knots(2, [0.5, 0.5, 0.75], 0.0, 1.0), 2)
        )
        s0, s1 = space.spaces

        def f_exact(x: Fraction, y: Fraction) -> Fraction:
            return x**3 * y**2 - x * y + 2 * y

        n0, n1 = int(s0.num_basis), int(s1.num_basis)
        g0, g1 = get_greville_abscissae(s0), get_greville_abscissae(s1)
        a0, a1 = _exact_collocation(s0, g0), _exact_collocation(s1, g1)
        data = [[f_exact(bspline_exact.exact(x), bspline_exact.exact(y)) for y in g1] for x in g0]
        c = _transpose(_solve_exact(a1, _transpose(_solve_exact(a0, data))))
        bound = _solve_bound(
            (2 * n0 + 2) + (2 * n1 + 2),
            _kappa_bound(a0) * _kappa_bound(a1),
            _frobenius(c),
            space.dtype,
        )

        def func(lat: PointsLattice) -> npt.NDArray[Any]:
            pts = lat.get_all_points()
            x, y = pts[:, 0], pts[:, 1]
            return x**3 * y**2 - x * y + 2 * y

        spline = interpolate_bspline(func, space)
        ctrl = np.asarray(spline.control_points)
        assert ctrl.shape == (n0, n1, 1)
        for i in range(n0):
            for j in range(n1):
                _assert_within(
                    Fraction(float(ctrl[i, j, 0])), c[i][j], bound, f"coefficient ({i}, {j})"
                )
        for x in _probe_points(0.0, 1.0):
            for y in (0.0, 0.31, 0.5, 0.97):
                want = f_exact(bspline_exact.exact(x), bspline_exact.exact(y))
                _assert_within(_exact_value(spline, [x, y])[0], want, bound, f"value at ({x}, {y})")

    def test_func_is_called_once_with_the_node_lattice_and_gets_the_shapes_it_expects(self) -> None:
        """Staged callable path: ``func`` counts its calls and checks the shapes it is given.

        Called exactly once, with a :class:`PointsLattice` of the node grid: for the default nodes
        the Greville abscissae, whose closed forms for uniform open knots are
        ``[0, 1/6, 1/2, 5/6, 1]``
        (degree 2, 3 intervals) and ``[0, 1/3, 2/3, 1]`` (degree 1, 3 intervals). The points come in
        C order (the last direction varies fastest). Bug caught: a second call (sampling once per
        direction or component), a lattice that is not the node grid, rows in F order.
        """
        space = create_uniform_space([2, 1], [3, 3])
        calls: list[PointsLattice] = []

        def func(lat: PointsLattice) -> npt.NDArray[Any]:
            calls.append(lat)
            shape = tuple(len(a) for a in lat.pts_per_dir)
            assert shape == (5, 4)
            pts = lat.get_all_points()
            assert pts.shape == (20, 2)
            assert pts.dtype == np.float64
            return np.stack([pts[:, 0] + pts[:, 1], pts[:, 0] * pts[:, 1]], axis=-1)

        spline = interpolate_bspline(func, space)

        assert len(calls) == 1
        assert isinstance(calls[0], PointsLattice)
        np.testing.assert_allclose(
            calls[0].pts_per_dir[0], [0.0, 1 / 6, 0.5, 5 / 6, 1.0], rtol=0.0, atol=4e-16
        )
        np.testing.assert_allclose(
            calls[0].pts_per_dir[1], [0.0, 1 / 3, 2 / 3, 1.0], rtol=0.0, atol=4e-16
        )
        pts = calls[0].get_all_points()
        assert pts[1, 0] == pts[0, 0]  # the last direction varies fastest
        assert pts[1, 1] != pts[0, 1]
        assert spline.rank == 2

    def test_custom_nodes_reach_the_function_as_given(self) -> None:
        """Staged callable path: the lattice ``func`` sees is the custom node grid, once."""
        space = create_uniform_space([2, 1], [3, 3])
        nodes = [np.linspace(0.0, 1.0, 9), np.array([0.0, 0.2, 0.5, 0.9, 1.0])]
        calls: list[PointsLattice] = []

        def func(lat: PointsLattice) -> npt.NDArray[Any]:
            calls.append(lat)
            return lat.get_all_points()[:, 0] + 2.0 * lat.get_all_points()[:, 1]

        spline = interpolate_bspline(func, space, nodes=nodes)

        assert len(calls) == 1
        for got, want in zip(calls[0].pts_per_dir, nodes, strict=True):
            np.testing.assert_array_equal(got, want)
        # x + 2y is bilinear, so lies in the space: the least-squares fit reproduces it.
        value = _exact_value(spline, [0.3, 0.7])[0]
        _assert_within(value, Fraction(3, 10) + 2 * Fraction(7, 10), 1e-12, "value at (0.3, 0.7)")


# ---------------------------------------------------------------------------
# fit_bspline
# ---------------------------------------------------------------------------


class TestFitTensorOracles:
    """Oracles for :func:`fit_bspline` on tensor-product nodes."""

    @pytest.mark.parametrize("n_nodes", [0, 20])
    def test_1d_fit_of_a_polynomial_is_exact_square_and_least_squares(self, n_nodes: int) -> None:
        """Oracle: an exactly reproduced polynomial and the exact (least-squares) solution.

        ``n_nodes = 0`` fits at the Greville abscissae (a square system), 20 gives an
        overdetermined one. The exact data are in the space, so both fits return its coefficients.
        Bug caught: an SVD truncation that drops a needed direction, the right-hand side not
        consistent with the nodes, a banded path that answers a rectangular system wrongly.
        """
        space = _space_from_knots((_knots(4, [0.125, 0.5, 0.75]), 4))
        s1 = space.spaces[0]
        poly = _dyadic_poly(4)
        nodes = get_greville_abscissae(s1) if n_nodes == 0 else np.linspace(0.0, 1.0, n_nodes)
        a = _exact_collocation(s1, nodes)
        c_exact = _lstsq_exact(a, [[_poly_exact(poly, bspline_exact.exact(x))] for x in nodes])
        bound = _solve_bound(
            len(nodes) + len(a[0]) + 2, _kappa_bound(a), _frobenius(c_exact), space.dtype
        )
        assert 1.0 / _kappa_bound(a) > 100 * float(np.finfo(np.float64).eps)  # truncation inactive

        spline = fit_bspline(_poly_float(poly, nodes), [nodes], space)

        ctrl = np.asarray(spline.control_points)
        for i in range(len(a[0])):
            _assert_within(Fraction(float(ctrl[i, 0])), c_exact[i][0], bound, f"coefficient {i}")
        for x in _probe_points(0.0, 1.0):
            _assert_within(
                _exact_value(spline, [x])[0],
                _poly_exact(poly, bspline_exact.exact(x)),
                bound,
                f"value at {x}",
            )

    @pytest.mark.parametrize("grid", [(0, 0), (12, 10)])
    def test_2d_fit_of_a_bipolynomial_square_and_least_squares(self, grid: tuple[int, int]) -> None:
        """Oracle: an exactly reproduced bi-polynomial and the exact Kronecker least squares.

        ``f(x, y) = x^2 y^2 + x - 3 y`` in degrees (3, 2), fitted at the Greville grid (square) and
        on a 12 x 10 uniform grid (overdetermined in both directions), given as a
        :class:`PointsLattice`. Bug caught: the directions' solves along the wrong axes (the sizes
        differ), a value array in F order.
        """
        space = _space_from_knots(
            (_knots(3, [0.25, 0.5, 0.75]), 3), (_knots(2, [0.5, 0.5, 0.75], 0.0, 1.0), 2)
        )
        s0, s1 = space.spaces
        nodes = [
            get_greville_abscissae(s) if m == 0 else np.linspace(0.0, 1.0, m)
            for s, m in zip((s0, s1), grid, strict=True)
        ]

        def f_exact(x: Fraction, y: Fraction) -> Fraction:
            return x**2 * y**2 + x - 3 * y

        a0, a1 = (_exact_collocation(s, nd) for s, nd in zip((s0, s1), nodes, strict=True))
        data = [
            [f_exact(bspline_exact.exact(x), bspline_exact.exact(y)) for y in nodes[1]]
            for x in nodes[0]
        ]
        c = _transpose(_lstsq_exact(a1, _transpose(_lstsq_exact(a0, data))))
        kappa = _kappa_bound(a0) * _kappa_bound(a1)
        sizes = len(a0) + len(a0[0]) + len(a1) + len(a1[0]) + 4
        bound = _solve_bound(sizes, kappa, _frobenius(c), space.dtype)

        xs, ys = np.meshgrid(nodes[0], nodes[1], indexing="ij")
        spline = fit_bspline(xs**2 * ys**2 + xs - 3 * ys, PointsLattice(nodes), space)

        ctrl = np.asarray(spline.control_points)
        for i, row in enumerate(c):
            for j, want in enumerate(row):
                _assert_within(
                    Fraction(float(ctrl[i, j, 0])), want, bound, f"coefficient ({i}, {j})"
                )
        for x in _probe_points(0.0, 1.0):
            for y in (0.0, 0.31, 0.97):
                _assert_within(
                    _exact_value(spline, [x, y])[0],
                    f_exact(bspline_exact.exact(x), bspline_exact.exact(y)),
                    bound,
                    f"value at ({x}, {y})",
                )

    def test_periodic_direction_wraps_in_the_fit(self) -> None:
        """Oracle: the exact solution of the wrapped collocation system.

        A periodic cubic space (6 intervals) fitted at its Greville abscissae to
        ``f(x) = 1 + x^3`` (not periodic, but any data have a unique interpolant): the coefficients
        are the exact solution of the exact wrapped system. Bug caught: a periodic wrap that is
        missing or shifted by one column, a border that is not moved back.
        """
        space = create_uniform_space(3, 6, periodic=True)
        s1 = space.spaces[0]
        nodes = get_greville_abscissae(s1)
        a = _exact_collocation(s1, nodes)
        poly = [Fraction(1), Fraction(0), Fraction(0), Fraction(1)]
        c_exact = _solve_exact(a, [[_poly_exact(poly, bspline_exact.exact(x))] for x in nodes])
        bound = _solve_bound(2 * len(nodes) + 2, _kappa_bound(a), _frobenius(c_exact), space.dtype)

        spline = fit_bspline(1.0 + nodes**3, [nodes], space)

        ctrl = np.asarray(spline.control_points)
        for i in range(len(nodes)):
            _assert_within(Fraction(float(ctrl[i, 0])), c_exact[i][0], bound, f"coefficient {i}")


class TestFitScatteredOracles:
    """Oracles for :func:`fit_bspline` on scattered nodes."""

    def test_scattered_fit_of_a_bipolynomial_is_exact(self) -> None:
        """Oracle: an exactly reproduced bi-polynomial and the exact normal-equations solution.

        Degrees (2, 2) on two intervals each (16 functions), 40 scattered points of the 1/64 grid
        (dyadic, so exact), ``f(x, y) = x^2 y - 2 x y^2 + y``. The exact collocation matrix is the
        row-wise Kronecker product of the exact 1D rows (first direction slowest). Bug caught: the
        two directions swapped in the row product, a transposed coefficient layout.
        """
        space = _space_from_knots((_knots(2, [0.5]), 2), (_knots(2, [0.25]), 2))
        s0, s1 = space.spaces
        rng = np.random.default_rng(20260101)
        pts = rng.integers(0, 65, size=(40, 2)).astype(np.float64) / 64.0

        def f_exact(x: Fraction, y: Fraction) -> Fraction:
            return x**2 * y - 2 * x * y**2 + y

        r0, r1 = _exact_collocation(s0, pts[:, 0]), _exact_collocation(s1, pts[:, 1])
        a = [[u * v for u in row0 for v in row1] for row0, row1 in zip(r0, r1, strict=True)]
        data = [[f_exact(bspline_exact.exact(x), bspline_exact.exact(y))] for x, y in pts]
        c = _lstsq_exact(a, data)
        kappa = _kappa_bound(a)
        assert 1.0 / kappa > 100 * float(np.finfo(np.float64).eps)  # truncation inactive
        bound = _solve_bound(len(a) + len(a[0]) + 2, kappa, _frobenius(c), space.dtype)

        spline = fit_bspline(
            pts[:, 0] ** 2 * pts[:, 1] - 2 * pts[:, 0] * pts[:, 1] ** 2 + pts[:, 1], pts, space
        )

        ctrl = np.asarray(spline.control_points)
        assert ctrl.shape == (4, 4, 1)
        for k, want in enumerate(c):
            _assert_within(
                Fraction(float(ctrl[k // 4, k % 4, 0])), want[0], bound, f"coefficient {k}"
            )
        for x in _probe_points(0.0, 1.0):
            for y in (0.0, 0.31, 0.97):
                _assert_within(
                    _exact_value(spline, [x, y])[0],
                    f_exact(bspline_exact.exact(x), bspline_exact.exact(y)),
                    bound,
                    f"value at ({x}, {y})",
                )

    def test_scattered_1d_square_system_is_the_interpolant(self) -> None:
        """Oracle: the exact solution of the square system, at the boundary of "underdetermined".

        As many points as functions (6) is solved, one fewer is refused. Bug caught: an off-by-one
        in the underdetermined check that refuses a solvable system or accepts an unsolvable one.
        """
        space = _space_from_knots((_knots(3, [0.5, 0.75]), 3))
        pts = np.array([0.0, 0.125, 0.375, 0.625, 0.875, 1.0])
        a = _exact_collocation(space.spaces[0], pts)
        poly = _dyadic_poly(3)
        c_exact = _solve_exact(a, [[_poly_exact(poly, bspline_exact.exact(x))] for x in pts])
        bound = _solve_bound(2 * 6 + 2, _kappa_bound(a), _frobenius(c_exact), space.dtype)

        spline = fit_bspline(_poly_float(poly, pts), pts.reshape(-1, 1), space)

        ctrl = np.asarray(spline.control_points)
        for i in range(6):
            _assert_within(Fraction(float(ctrl[i, 0])), c_exact[i][0], bound, f"coefficient {i}")
        with pytest.raises(
            ValueError, match=r"Underdetermined system: 5 points < 6 basis functions"
        ):
            fit_bspline(_poly_float(poly, pts[:5]), pts[:5].reshape(-1, 1), space)


class TestFitTruncation:
    """The default truncation tolerance of :func:`fit_bspline`, on a system that needs it."""

    @pytest.mark.parametrize("layout", ["tensor", "scattered"])
    def test_a_rank_deficient_fit_is_the_exact_minimum_norm_solution(self, layout: str) -> None:
        """Oracle: the exact minimum-norm solution ``A_u^T (A_u A_u^T)^-1 b_u`` in rationals.

        Degree 2 on five equal intervals (7 functions) and 7 nodes of which only 5 are distinct
        (0 three times), so the exact collocation matrix has rank 5 and two of its computed
        singular values are rounding noise. The data are a polynomial of the space, so the system
        is consistent; its minimum-norm solution is the one the truncated pseudo-inverse returns,
        computed here from the distinct rows (a full-row-rank matrix) by the normal equations,
        exactly. The repeated rows make the LU meet an exactly zero pivot, which sends the square
        system to the truncated SVD. The bound is the file's ``_solve_bound``, with ``kappa`` that
        of the five distinct rows, which bounds ``s_max / s_min`` of the retained values: the
        assertion before the call checks that they lie above the default cutoff
        ``100 eps s_max``. What is pinned is that a default cutoff exists and keeps the retained
        values; the factor 100 itself is not pinned (a smaller one passes while it stays above the
        noise).

        Bug caught: the default tolerance not reaching the solve (a zero cutoff), which inverts the
        two noise-level singular values and inflates the coefficients by the reciprocal of the
        noise. The scattered route resolves the default on its own, so both layouts are run.
        """
        space = create_uniform_space(2, 5)
        poly = _dyadic_poly(2)
        nodes = np.array([0.0, 0.0, 0.0, 0.3, 0.5, 0.7, 1.0])
        distinct = np.unique(nodes)
        assert len(distinct) == 5
        a_u = _exact_collocation(space.spaces[0], distinct)
        a_ut = _transpose(a_u)
        b_u = [[_poly_exact(poly, bspline_exact.exact(x))] for x in distinct]
        c_exact = _matmul(a_ut, _solve_exact(_matmul(a_u, a_ut), b_u))
        bound = _solve_bound(
            len(nodes) + len(a_u[0]) + 2, _kappa_bound(a_ut), _frobenius(c_exact), space.dtype
        )
        assert 1.0 / _kappa_bound(a_ut) > 100 * float(np.finfo(np.float64).eps)  # kept values kept

        where: Any = [nodes] if layout == "tensor" else nodes.reshape(-1, 1)
        spline = fit_bspline(_poly_float(poly, nodes), where, space)

        ctrl = np.asarray(spline.control_points)
        assert ctrl.shape == (7, 1)
        for i in range(7):
            _assert_within(Fraction(float(ctrl[i, 0])), c_exact[i][0], bound, f"coefficient {i}")


# ---------------------------------------------------------------------------
# l2_project_bspline
# ---------------------------------------------------------------------------


def _quadrature_weights(nodes: Sequence[Fraction]) -> list[Fraction]:
    """Return the weights on [0, 1] of the interpolatory rule on ``nodes`` (moment equations)."""
    m = len(nodes)
    vandermonde = [[t**j for t in nodes] for j in range(m)]
    moments = [[Fraction(1, j + 1)] for j in range(m)]
    return [row[0] for row in _solve_exact(vandermonde, moments)]


def _local_basis(knots: Sequence[Fraction], p: int, span: int, x: Fraction) -> list[Fraction]:
    """Return ``N_{span-p+a, p}(x)``, ``a = 0 .. p``, by the recursive definition on one span.

    The Cox-de Boor recursion in exact arithmetic, evaluated on the polynomial piece of the given
    span (so ``x`` may be any rational, not only a float), a term with a zero denominator being 0.

    Args:
        knots (Sequence[Fraction]): The exact knot vector.
        p (int): The degree.
        span (int): The span index ``j`` with ``knots[j] <= x < knots[j + 1]``.
        x (Fraction): The point.

    Returns:
        list[Fraction]: The ``p + 1`` functions that are non-zero on the span.
    """
    memo: dict[tuple[int, int], Fraction] = {}

    def basis(i: int, q: int) -> Fraction:
        if (i, q) not in memo:
            if q == 0:
                memo[i, q] = Fraction(int(i == span))
            else:
                value = Fraction(0)
                d1 = knots[i + q] - knots[i]
                if d1 != 0:
                    value += (x - knots[i]) / d1 * basis(i, q - 1)
                d2 = knots[i + q + 1] - knots[i + 1]
                if d2 != 0:
                    value += (knots[i + q + 1] - x) / d2 * basis(i + 1, q - 1)
                memo[i, q] = value
        return memo[i, q]

    return [basis(span - p + a, p) for a in range(p + 1)]


def _gram_and_load(space: BsplineSpace1D, f: Sequence[Fraction]) -> tuple[Exact, Exact]:
    """Return the exact Gram matrix and the load of a polynomial ``f`` in a 1D space.

    Span by span, the interpolatory rule on ``2 p + 2`` interior equispaced nodes, exact for the
    degree ``2 p + 1`` of ``N_i f`` when ``f`` has degree ``p + 1`` (and for the degree ``2 p`` of
    ``N_i N_j``). The functions of a periodic space wrap.

    Args:
        space (BsplineSpace1D): The space.
        f (Sequence[Fraction]): The polynomial's coefficients, degree at most ``p + 1``.

    Returns:
        tuple[Exact, Exact]: ``(G, b)`` with ``b`` of extents ``(n, 1)``.
    """
    p = space.degree
    n = int(space.num_basis)
    knots = np.asarray(space.knots)
    m = 2 * p + 2
    unit_nodes = [Fraction(k + 1, m + 1) for k in range(m)]
    weights = _quadrature_weights(unit_nodes)
    exact_knots = [bspline_exact.exact(k) for k in knots]
    lo, hi = (bspline_exact.exact(v) for v in space.domain)
    gram = [[Fraction(0)] * n for _ in range(n)]
    load = [[Fraction(0)] for _ in range(n)]
    for j in range(len(knots) - 1):
        u0, u1 = bspline_exact.exact(knots[j]), bspline_exact.exact(knots[j + 1])
        if not (u0 < u1 and lo <= u0 and u1 <= hi):
            continue
        for t, w in zip(unit_nodes, weights, strict=True):
            x = u0 + (u1 - u0) * t
            vals = _local_basis(exact_knots, p, j, x)
            first = j - p
            fx = _poly_exact(f, x)
            for a_, va in enumerate(vals):
                load[(first + a_) % n][0] += (u1 - u0) * w * va * fx
                for b_, vb in enumerate(vals):
                    gram[(first + a_) % n][(first + b_) % n] += (u1 - u0) * w * va * vb
    return gram, load


def _pin_ends(
    gram: Exact, load: Exact, f: Sequence[Fraction], space: BsplineSpace1D
) -> tuple[Exact, Exact]:
    """Replace the first and last equations by the basis at the ends (boundary interpolation)."""
    n = len(gram)
    a = _exact_collocation(space, np.array([space.domain[0], space.domain[1]]))
    gram = [list(r) for r in gram]
    load = [list(r) for r in load]
    gram[0], gram[-1] = a[0], a[1]
    load[0] = [_poly_exact(f, bspline_exact.exact(space.domain[0]))]
    load[-1] = [_poly_exact(f, bspline_exact.exact(space.domain[1]))]
    assert len(gram) == n
    return gram, load


class TestL2ProjectOracles:
    """Oracles for :func:`l2_project_bspline`."""

    @pytest.mark.parametrize(
        ("name", "kwargs"),
        [
            ("uniform cubic, 4 intervals", {}),
            ("nonuniform quartic", {}),
            ("quadratic, C0 at a double knot", {}),
            ("cubic on [2, 5]", {}),
            ("float32 cubic", {}),
            ("uniform cubic, 4 intervals", {"quadrature": "gauss-lobatto", "n_quad": 6}),
        ],
    )
    def test_projection_of_a_polynomial_outside_the_space_is_G_inverse_b(
        self, name: str, kwargs: dict[str, Any]
    ) -> None:
        """Oracle: exact-rational Gram matrix and load.

        ``f`` has degree ``p + 1``, so it is not in the space (the projection is not an
        interpolant) and the library's ``p + 1`` Gauss points per span integrate ``N_i f`` exactly;
        the Lobatto variant uses 6 points per span, exact to degree 9 >= 2 p + 1. Expected
        coefficients: ``G^-1 b`` with ``G`` and ``b`` from interior-node interpolatory quadrature
        in rationals. Bug caught: a load missing the span length, a mass matrix that is not the Gram
        matrix, a Lobatto rule applied with the wrong weights.
        """
        space = _SPACES_1D[name]()
        s1 = space.spaces[0]
        f = _dyadic_poly(s1.degree + 1)
        gram, load = _gram_and_load(s1, f)
        c_exact = _solve_exact(gram, load)
        n = int(s1.num_basis)
        n_quad = kwargs.get("n_quad", s1.degree + 1)
        points = n_quad * int(s1.num_intervals)
        bound = _solve_bound(
            2 * n + 2 + points, _kappa_bound(gram), _frobenius(c_exact), space.dtype
        )

        spline = l2_project_bspline(
            lambda lat: _poly_float(f, lat.get_all_points()[:, 0]), space, **kwargs
        )

        ctrl = np.asarray(spline.control_points)
        for i in range(n):
            _assert_within(Fraction(float(ctrl[i, 0])), c_exact[i][0], bound, f"coefficient {i}")

    def test_periodic_projection_wraps_the_gram_matrix(self) -> None:
        """Oracle: exact-rational Gram matrix and load of the wrapped functions.

        The periodic functions are the sums of the unwrapped ones that share an index mod ``n``;
        ``f = x^4 - x/3`` restricted to ``[0, 1]`` is projected. Bug caught: a mass matrix or load
        assembled without the wrap, a wrap to the wrong column.
        """
        space = create_uniform_space(3, 6, periodic=True)
        s1 = space.spaces[0]
        f = [Fraction(0), Fraction(-1, 3), Fraction(0), Fraction(0), Fraction(1)]
        gram, load = _gram_and_load(s1, f)
        c_exact = _solve_exact(gram, load)
        n = int(s1.num_basis)
        bound = _solve_bound(
            2 * n + 2 + 4 * 6, _kappa_bound(gram), _frobenius(c_exact), space.dtype
        )

        spline = l2_project_bspline(lambda lat: _poly_float(f, lat.get_all_points()[:, 0]), space)

        ctrl = np.asarray(spline.control_points)
        for i in range(n):
            _assert_within(Fraction(float(ctrl[i, 0])), c_exact[i][0], bound, f"coefficient {i}")

    def test_boundary_interpolation_replaces_the_end_equations_by_point_values(self) -> None:
        """Oracle: the exact system with its first and last rows replaced by the basis at the ends.

        With ``boundary_interpolation=True`` the first and last equations are ``N(end) c = f(end)``
        instead of ``G c = b``; for a clamped space the rows are unit vectors. Expected: the exact
        solution of that modified rational system. Bug caught: the replaced right-hand side taken as
        the load integral instead of the point value, the end rows swapped.
        """
        space = _space_from_knots((_knots(3, [0.25, 0.5, 0.75]), 3))
        s1 = space.spaces[0]
        f = _dyadic_poly(4)
        gram, load = _gram_and_load(s1, f)
        gram, load = _pin_ends(gram, load, f, s1)
        c_exact = _solve_exact(gram, load)
        n = int(s1.num_basis)
        bound = _solve_bound(2 * n + 2 + 16, _kappa_bound(gram), _frobenius(c_exact), space.dtype)

        spline = l2_project_bspline(
            lambda lat: _poly_float(f, lat.get_all_points()[:, 0]),
            space,
            boundary_interpolation=True,
        )

        ctrl = np.asarray(spline.control_points)
        for i in range(n):
            _assert_within(Fraction(float(ctrl[i, 0])), c_exact[i][0], bound, f"coefficient {i}")
        # The ends are matched to the point value, exactly (a unit row): f(0) and f(1).
        _assert_within(
            _exact_value(spline, [0.0])[0], _poly_exact(f, Fraction(0)), bound, "value at 0"
        )
        _assert_within(
            _exact_value(spline, [1.0])[0], _poly_exact(f, Fraction(1)), bound, "value at 1"
        )

    @pytest.mark.parametrize("pinned", [False, True])
    def test_separable_function_on_a_tensor_space_is_an_outer_product(self, pinned: bool) -> None:
        """Oracle: for ``f(x, y) = f0(x) f1(y)`` the projection is ``c0 (x) c1``, exactly.

        The Kronecker system ``(G0 (x) G1) c = b0 (x) b1`` is solved by ``c = c0 (x) c1`` with
        ``c_d`` the exact 1D projections. With ``pinned`` the end rows of each direction are
        replaced (faces and corners both enter): the right-hand side of a replaced row is then the
        outer product of the point value and the other direction's load, which is the face
        integral of the trace, and the corner is the product of the two point values. Bug caught:
        the faces and corners applied in the wrong order or at the wrong end, a trace load with the
        wrong axis contracted.
        """
        space = _space_from_knots(
            (_knots(3, [0.25, 0.5, 0.75]), 3), (_knots(2, [0.5], 0.0, 1.0), 2)
        )
        s0, s1 = space.spaces
        f0, f1 = _dyadic_poly(4), _dyadic_poly(3, salt=3)
        parts = []
        for s, f in ((s0, f0), (s1, f1)):
            gram, load = _gram_and_load(s, f)
            if pinned:
                gram, load = _pin_ends(gram, load, f, s)
            parts.append((gram, _solve_exact(gram, load)))
        (g0, c0), (g1, c1) = parts
        kappa = _kappa_bound(g0) * _kappa_bound(g1)
        c_norm = _frobenius(c0) * _frobenius(c1)
        sizes = (2 * len(g0) + 2 + 16) + (2 * len(g1) + 2 + 12)
        bound = _solve_bound(sizes, kappa, c_norm, space.dtype)

        def func(lat: PointsLattice) -> npt.NDArray[Any]:
            pts = lat.get_all_points()
            return _poly_float(f0, pts[:, 0]) * _poly_float(f1, pts[:, 1])

        spline = l2_project_bspline(func, space, boundary_interpolation=pinned)

        ctrl = np.asarray(spline.control_points)
        assert ctrl.shape == (len(g0), len(g1), 1)
        for i in range(len(g0)):
            for j in range(len(g1)):
                _assert_within(
                    Fraction(float(ctrl[i, j, 0])),
                    c0[i][0] * c1[j][0],
                    bound,
                    f"coefficient ({i}, {j})",
                )

    def test_func_call_pattern_with_boundary_interpolation_in_2d(self) -> None:
        """Staged callable path: the previous release's call pattern, kept on purpose.

        A 2D vector-valued function with ``boundary_interpolation=True`` is called ``1 + 8 n``
        times for ``n`` components: once on the quadrature lattice, then for each component the
        four faces and the four corners in the order of the traces, with lattice extents
        ``(1, nq1), (1, nq1), (nq0, 1), (nq0, 1)`` and ``(1, 1)`` four times. Every call is given a
        lattice and returns ``(points, n)``. Bug caught: a changed sampling order (a trace written
        over a later one), faces sampled once for all components, a lattice of the wrong extents.
        """
        space = create_uniform_space([3, 2], [4, 2])
        nq0, nq1 = 4 * 4, 3 * 2  # degree + 1 Gauss points on each of the intervals
        n_components = 3
        shapes: list[tuple[int, ...]] = []

        def func(lat: PointsLattice) -> npt.NDArray[Any]:
            assert isinstance(lat, PointsLattice)
            shape = tuple(len(a) for a in lat.pts_per_dir)
            shapes.append(shape)
            pts = lat.get_all_points()
            assert pts.shape == (int(np.prod(shape)), 2)
            return np.stack([pts[:, 0], pts[:, 1], pts[:, 0] * pts[:, 1]], axis=-1)

        l2_project_bspline(func, space, boundary_interpolation=True)

        per_component = [(1, nq1), (1, nq1), (nq0, 1), (nq0, 1), (1, 1), (1, 1), (1, 1), (1, 1)]
        assert len(shapes) == 1 + 8 * n_components
        assert shapes == [(nq0, nq1), *(per_component * n_components)]

    def test_func_is_called_once_without_boundary_interpolation(self) -> None:
        """Staged callable path: a single call on the quadrature lattice, whatever the rank."""
        space = create_uniform_space([2, 2], [3, 2])
        shapes: list[tuple[int, ...]] = []

        def func(lat: PointsLattice) -> npt.NDArray[Any]:
            shapes.append(tuple(len(a) for a in lat.pts_per_dir))
            pts = lat.get_all_points()
            return np.stack([pts[:, 0], pts[:, 1]], axis=-1)

        l2_project_bspline(func, space)

        assert shapes == [(9, 6)]


# ---------------------------------------------------------------------------
# quasi_interpolate_bspline
# ---------------------------------------------------------------------------


def _dyadic_ctrl(
    shape: tuple[int, ...], seed: int, dtype: npt.DTypeLike = np.float64
) -> npt.NDArray[Any]:
    """Return control points that are multiples of 1/64 in [-1, 1] (exact in float32)."""
    rng = np.random.default_rng(seed)
    return (rng.integers(-64, 65, size=shape) / 64.0).astype(dtype)


def _record_points(space: BsplineSpace) -> npt.NDArray[np.float64]:
    """Return the sample points of the quasi-interpolant of a space (recorded from its call)."""
    seen: list[npt.NDArray[np.float64]] = []

    def func(pts: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        seen.append(pts.copy())
        return np.zeros(len(pts))

    quasi_interpolate_bspline(func, space)
    return seen[0]


def _local_block_kappa(space1d: BsplineSpace1D) -> float:
    """Return the largest kappa bound of the local collocation blocks of a direction.

    The block of a function is the exact basis of the ``p + 1`` functions supported on its interval
    at its ``p + 1`` sample points (read from the quasi-interpolant's own points).
    """
    p = space1d.degree
    pts = _record_points(BsplineSpace([space1d]))[:, 0]
    knots = np.asarray(space1d.knots)
    worst = 1.0
    for j in range(int(space1d.num_basis)):
        block: Exact = []
        for x in pts[j * (p + 1) : (j + 1) * (p + 1)]:
            basis = bspline_exact.ExactBasis(knots, p, x)
            block.append(basis.local())
        worst = max(worst, _kappa_bound(block))
    return worst


def _qi_bound(space: BsplineSpace, c_inf: float) -> float:
    """Return the bound of ``Q B - B`` in the coefficients (see the module docstring)."""
    dtype = space.dtype
    u_double = float(np.finfo(np.float64).eps) / 2.0
    u_space = float(np.finfo(dtype).eps) / 2.0
    kappa = 1.0
    orders = 0.0
    basis_units = 0.0
    for s in space.spaces:
        kappa *= _local_block_kappa(s)
        orders += s.degree + 1.0
        basis_units += (s.degree + 1.0) * (6.0 * s.degree + 1.0)
    # The inverse of each local block is computed in double (backward stable); the block itself
    # has the roundings of the triangle in the space's own type, (6 p + 1) u_T per entry.
    return (K_SLACK * (orders + 2.0) * u_double + basis_units * u_space) * kappa * max(c_inf, 1.0)


def _exact_sampler(
    space: BsplineSpace, ctrl: npt.NDArray[Any]
) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
    """Return a function of ``(M, dim)`` points giving the exact values of the spline ``ctrl``."""
    spaces = space.spaces
    knots = [s.knots for s in spaces]
    degrees = [s.degree for s in spaces]

    def func(pts: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        zeros = [0] * len(spaces)
        rows = [bspline_exact.homogeneous(knots, degrees, ctrl, list(x), zeros) for x in pts]
        return bspline_exact.to_float_array(rows)

    return func


_QI_CASES: dict[str, Callable[[], tuple[BsplineSpace, tuple[int, ...]]]] = {
    "uniform cubic": lambda: (create_uniform_space(3, 5), (1,)),
    "cubic with a double knot, rank 2": lambda: (
        _space_from_knots((_knots(3, [0.25, 0.5, 0.5, 0.75]), 3)),
        (2,),
    ),
    "periodic cubic, 8 intervals": lambda: (create_uniform_space(3, 8, periodic=True), (1,)),
    "float32 cubic": lambda: (create_uniform_space(3, 5, dtype=np.float32), (1,)),
    "2D, degrees (2, 3), rank 2": lambda: (
        _space_from_knots((_knots(2, [0.25, 0.5, 0.5]), 2), (_knots(3, [0.5]), 3)),
        (2,),
    ),
}


class TestQuasiInterpolateOracles:
    """Oracles for :func:`quasi_interpolate_bspline`."""

    @pytest.mark.parametrize("name", list(_QI_CASES))
    def test_a_random_spline_of_the_space_is_reproduced(self, name: str) -> None:
        """Oracle: the invariant ``Q B = B``, with the exact values of ``B`` as the samples.

        A spline of the space with dyadic control points is sampled in exact arithmetic (the
        recursive definition) at the points ``func`` is given; the quasi-interpolant of that
        function must return the control points. The tolerance uses the exact condition number of
        the local blocks. Bug caught: sample points outside the function's interval, weights of
        another function's row, a contraction in the wrong direction order, a periodic function
        wrapped to the wrong index.
        """
        space, rank_shape = _QI_CASES[name]()
        shape = tuple(int(n) for n in space.num_basis) + rank_shape
        ctrl = _dyadic_ctrl(shape, seed=len(name), dtype=space.dtype)
        exact_values = _exact_sampler(space, ctrl.astype(np.float64))

        spline = quasi_interpolate_bspline(exact_values, space)

        got = np.asarray(spline.control_points)
        assert got.shape == ctrl.shape
        assert got.dtype == space.dtype
        bound = _qi_bound(space, float(np.max(np.abs(ctrl))))
        worst = float(np.max(np.abs(got.astype(np.float64) - ctrl.astype(np.float64))))
        assert worst <= bound, f"max |QB - B| = {worst} > {bound}"

    def test_a_polynomial_of_the_space_degree_is_reproduced(self) -> None:
        """Oracle: an exactly reproduced polynomial (the cubic ``x^3 - x + 1/2``), exact evaluation.

        Bug caught: a quasi-interpolant that reproduces random splines but not the smooth
        functions the space contains (a wrong block orientation cancelled by symmetric data).
        """
        space = create_uniform_space(3, 5)
        poly = [Fraction(1, 2), Fraction(-1), Fraction(0), Fraction(1)]
        spline = quasi_interpolate_bspline(lambda pts: _poly_float(poly, pts[:, 0]), space)
        bound = _qi_bound(space, 1.0)
        for x in _probe_points(0.0, 1.0):
            _assert_within(
                _exact_value(spline, [x])[0],
                _poly_exact(poly, bspline_exact.exact(x)),
                bound,
                f"value at {x}",
            )

    def test_func_is_called_once_with_a_float64_c_contiguous_point_array(self) -> None:
        """Staged callable path: one call, one ``(M, dim)`` float64 C-contiguous array.

        ``M`` is the number of functions times the points per function, ``prod(num_basis) *
        prod(degree + 1)``: 4 x 3 functions with 3 x 2 points for the space below. Bug caught: one
        call per function or per component, a Fortran-ordered or non-float64 array handed to a
        user function, the wrong extent.
        """
        space = _space_from_knots((_knots(2, [0.5]), 2), (_knots(1, []), 1))
        calls: list[npt.NDArray[Any]] = []

        def func(pts: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            calls.append(pts)
            assert isinstance(pts, np.ndarray)
            assert pts.dtype == np.float64
            assert pts.flags["C_CONTIGUOUS"]
            assert pts.ndim == 2
            return np.stack([pts[:, 0], pts[:, 1]], axis=-1)

        spline = quasi_interpolate_bspline(func, space)

        assert len(calls) == 1
        assert calls[0].shape == (4 * 2 * 3 * 2, 2)
        assert spline.rank == 2

    def test_scalar_and_single_column_functions_agree(self) -> None:
        """Staged callable path: ``(M,)`` and ``(M, 1)`` answers give the same coefficients."""
        space = create_uniform_space(3, 4)

        def column(pts: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return np.sin(3.0 * pts[:, :1])

        a = quasi_interpolate_bspline(lambda pts: column(pts)[:, 0], space)
        b = quasi_interpolate_bspline(column, space)
        np.testing.assert_array_equal(np.asarray(a.control_points), np.asarray(b.control_points))


# ---------------------------------------------------------------------------
# dtype and argument decisions
# ---------------------------------------------------------------------------


class TestDtypeDecisions:
    """The decisions the previous release made, probed against it and kept (type and message)."""

    @staticmethod
    def _cubic() -> tuple[BsplineSpace, npt.NDArray[Any]]:
        space = create_uniform_space(3, 4)  # 7 functions
        return space, get_greville_abscissae(space.spaces[0])

    def test_float16_values_on_a_square_system_hit_linalg(self) -> None:
        """A float16 solve was refused by ``numpy.linalg.solve``: a TypeError with its message."""
        space, g = self._cubic()
        with pytest.raises(TypeError, match=r"array type float16 is unsupported in linalg"):
            fit_bspline((g**2).astype(np.float16), [g], space)

    def test_float16_values_on_an_overdetermined_grid_hit_the_space_dtype_check(self) -> None:
        """The SVD path promoted to float64 and the constructor refused the float16 result."""
        space, _ = self._cubic()
        nodes = np.linspace(0.0, 1.0, 20)
        with pytest.raises(ValueError, match=r"same dtype as the B-spline space"):
            fit_bspline((nodes**2).astype(np.float16), [nodes], space)

    def test_float32_values_on_a_float64_space_are_refused(self) -> None:
        """The control points must have the space's dtype, for a square and a rectangular system."""
        space, g = self._cubic()
        nodes = np.linspace(0.0, 1.0, 20)
        for x in (g, nodes):
            with pytest.raises(ValueError, match=r"same dtype as the B-spline space"):
                fit_bspline((x**2).astype(np.float32), [x], space)

    @pytest.mark.parametrize("n_nodes", [0, 20])
    def test_byte_swapped_and_masked_values_fit_like_native_float64(self, n_nodes: int) -> None:
        """Byte-swapped ``>f8`` values and masked arrays are solved as the native float64 array.

        The conversion happens before the solve, so the same numbers go through the same code:
        the coefficients are identical, not merely close.
        """
        space, g = self._cubic()
        nodes = g if n_nodes == 0 else np.linspace(0.0, 1.0, n_nodes)
        native = nodes**2 + 0.25
        want = np.asarray(fit_bspline(native, [nodes], space).control_points)
        swapped = native.astype(">f8")
        assert not swapped.dtype.isnative
        np.testing.assert_array_equal(
            np.asarray(fit_bspline(swapped, [nodes], space).control_points), want
        )
        mask = np.zeros(len(nodes), dtype=bool)
        mask[1] = True  # the mask is ignored: the underlying data are fitted
        masked = np.ma.masked_array(native, mask=mask)
        np.testing.assert_array_equal(
            np.asarray(fit_bspline(masked, [nodes], space).control_points), want
        )

    def test_integer_scattered_nodes_need_an_explicit_tolerance(self) -> None:
        """Integer nodes reached the default tolerance's machine epsilon: ValueError "not inexact".

        With an explicit ``tol`` the fit is valid. Oracle for it: degree-1 hats on the knots
        ``0 .. 4`` interpolated at the integer nodes ``0 .. 4`` are the data themselves
        (``N_i(j) = delta_ij``), here ``j^2``.
        """
        space = _space_from_knots((_knots(1, [1.0, 2.0, 3.0], 0.0, 4.0), 1))
        pts = np.arange(5).reshape(-1, 1)
        values = (np.arange(5) ** 2).astype(np.float64)
        with pytest.raises(ValueError, match=r"not inexact"):
            fit_bspline(values, pts, space)  # type: ignore[arg-type]
        spline = fit_bspline(values, pts, space, tol=1e-12)  # type: ignore[arg-type]
        np.testing.assert_allclose(
            np.asarray(spline.control_points)[:, 0], values, rtol=0.0, atol=1e-12
        )

    def test_boundary_derivative_counts_of_the_wrong_type_are_refused_like_before(self) -> None:
        """A positive float or bool count is a TypeError, a triple a ValueError (as before)."""
        space = create_uniform_space(3, 4)

        def func(lat: PointsLattice) -> npt.NDArray[Any]:
            return lat.get_all_points()[:, 0] ** 2

        with pytest.raises(TypeError, match=r"integer"):
            interpolate_bspline(func, space, boundary_derivatives=[(1.0, 1)])  # type: ignore[list-item]
        with pytest.raises(TypeError, match=r"integer is required"):
            interpolate_bspline(func, space, boundary_derivatives=[(True, 0)])
        with pytest.raises(ValueError, match=r"too many values to unpack"):
            interpolate_bspline(func, space, boundary_derivatives=[(1, 1, 1)])

    def test_more_derivative_rows_than_nodes_allow_raise_numpys_broadcast_message(self) -> None:
        """Seven rows into the six that fit between the first and the last of 7 nodes.

        The boundary is constructed: six rows fit (the whole interior of the 7 nodes), seven do
        not. The message is NumPy's, with the shapes ``(7,7)`` into ``(6,7)``, at either end.
        """
        space = create_uniform_space(3, 4)
        calls: list[int] = []

        def func(lat: PointsLattice) -> npt.NDArray[Any]:
            calls.append(1)
            return lat.get_all_points()[:, 0] ** 2

        interpolate_bspline(func, space, boundary_derivatives=[(6, 0)])
        interpolate_bspline(func, space, boundary_derivatives=[(0, 6)])
        pattern = r"could not broadcast input array from shape \(7,7\) into shape \(6,7\)"
        with pytest.raises(ValueError, match=pattern):
            interpolate_bspline(func, space, boundary_derivatives=[(7, 0)])
        with pytest.raises(ValueError, match=pattern):
            interpolate_bspline(func, space, boundary_derivatives=[(0, 7)])
        assert len(calls) == 2  # the system is built, and refused, before the function is sampled

    def test_a_degenerate_end_span_raises_zero_division_error(self) -> None:
        """The derivative recurrence's zero divisor reaches Python as ZeroDivisionError.

        Knots ``0 0 0 1 2 2 3 4`` (degree 2): the domain ends at 2, whose clamped last span is
        empty, so the order-2 row at the right end divides by zero; order 1 does not.
        """
        space = BsplineSpace([BsplineSpace1D(np.array([0, 0, 0, 1, 2, 2, 3, 4.0]), 2)])
        nodes = np.array([0.0, 0.5, 1.0, 1.5, 2.0])

        def func(lat: PointsLattice) -> npt.NDArray[Any]:
            return lat.get_all_points()[:, 0]

        with pytest.raises(ZeroDivisionError, match=re.escape("division by zero")):
            interpolate_bspline(func, space, nodes=nodes, boundary_derivatives=[(0, 2)])
        interpolate_bspline(func, space, nodes=nodes, boundary_derivatives=[(0, 1)])


class TestInputLayouts:
    """Memory layout and element type of the inputs do not change the fit."""

    def test_fortran_strided_read_only_integer_and_list_values_fit_like_contiguous_floats(
        self,
    ) -> None:
        """Layout cases: the same numbers in another layout go through the same solve.

        Values as a Fortran-ordered array, a strided view, a read-only array, a nested list and an
        ``int64`` array (of the same numbers), and nodes as a strided view, give the coefficients
        of the contiguous float64 fit, identical rather than close: the wrapper converts to
        contiguous float64 before the compiled solve, so equal numbers meet equal code. Bug caught:
        a compiled routine reading a non-contiguous buffer as if it were contiguous (the
        ``noconvert`` arguments make that a TypeError or, worse, a wrong read).
        """
        space = create_uniform_space([3, 2], [4, 2])  # 7 x 4 functions
        g0 = get_greville_abscissae(space.spaces[0])
        g1 = get_greville_abscissae(space.spaces[1])
        values = np.round(np.outer(g0**2, 1.0 + g1) * 64.0) / 64.0  # multiples of 1/64
        want = np.asarray(fit_bspline(values, [g0, g1], space).control_points)

        fortran = np.asfortranarray(values)
        padded = np.zeros((14, 4))
        padded[::2] = values
        read_only = values.copy()
        read_only.flags.writeable = False
        strided_nodes = np.repeat(g0, 2)[::2]
        assert not fortran.flags.c_contiguous
        assert not padded[::2].flags.c_contiguous
        assert not strided_nodes.flags.c_contiguous
        cases: list[tuple[Any, list[npt.NDArray[Any]]]] = [
            (fortran, [g0, g1]),
            (padded[::2], [g0, g1]),
            (read_only, [g0, g1]),
            (values.tolist(), [g0, g1]),
            ((values * 64.0).astype(np.int64) / 64.0, [g0, g1]),
            (values, [strided_nodes, g1]),
        ]
        for vals, nodes in cases:
            got = np.asarray(fit_bspline(vals, nodes, space).control_points)
            np.testing.assert_array_equal(got, want)
        # Integer values themselves are converted to float64 first.
        ints = np.round(values * 64.0).astype(np.int64)
        got_int = np.asarray(fit_bspline(ints, [g0, g1], space).control_points)
        np.testing.assert_array_equal(
            got_int,
            np.asarray(fit_bspline(ints.astype(np.float64), [g0, g1], space).control_points),
        )

    def test_integer_valued_function_is_converted_to_float64(self) -> None:
        """Oracle: hat functions interpolate their data (``N_i(g_j) = delta_ij``).

        On degree 1 with 4 uniform intervals the Greville abscissae are the knots, so the
        coefficients are the samples themselves: here the integers 0 .. 4 an ``int64`` function
        returns. Bug caught: an integer array handed to the solver unconverted (a dtype error), or
        coefficients in an integer dtype.
        """
        space = create_uniform_space(1, 4)
        spline = interpolate_bspline(lambda lat: np.arange(len(lat.get_all_points())), space)
        ctrl = np.asarray(spline.control_points)
        assert ctrl.dtype == np.float64
        np.testing.assert_allclose(ctrl[:, 0], np.arange(5.0), rtol=0.0, atol=1e-14)


class TestKnownDefects:
    """Defects found while writing the oracles above, pinned as expected failures.

    Both are in the algorithm and are shared with the previous release (probed against it).
    """

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Q B != B for some periodic spaces (cubic with 6 or 12 intervals; 5, 7, 8, 9, 10 "
            "pass): the abscissa of function j is read from the wrapped and sorted Greville array "
            "and the block row is taken in the unwrapped numbering (the previous release does the "
            "same)"
        ),
    )
    def test_quasi_interpolant_reproduces_a_periodic_cubic_with_six_intervals(self) -> None:
        """Oracle: the invariant ``Q B = B`` on ``create_uniform_space(3, 6, periodic=True)``.

        The coefficients come back cyclically shifted, an error of order 1.
        """
        space = create_uniform_space(3, 6, periodic=True)
        ctrl = _dyadic_ctrl((6, 1), seed=1)
        spline = quasi_interpolate_bspline(_exact_sampler(space, ctrl), space)
        got = np.asarray(spline.control_points)
        assert float(np.max(np.abs(got - ctrl))) <= _qi_bound(space, float(np.max(np.abs(ctrl))))

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Q B != B for a knot of multiplicity degree + 1: the Greville abscissa of the function "
            "that ends at the knot picks the interval to its right, outside the function's "
            "support, and the block row is indexed from the end (the previous release does the "
            "same)"
        ),
    )
    def test_quasi_interpolant_reproduces_a_spline_with_a_discontinuity(self) -> None:
        """Oracle: the invariant ``Q B = B`` on a cubic with the interior knot 1/2 four times."""
        space = _space_from_knots((_knots(3, [0.25, 0.5, 0.5, 0.5, 0.5, 0.75]), 3))
        ctrl = _dyadic_ctrl((int(space.num_basis[0]), 1), seed=3)
        spline = quasi_interpolate_bspline(_exact_sampler(space, ctrl), space)
        got = np.asarray(spline.control_points)
        assert float(np.max(np.abs(got - ctrl))) <= _qi_bound(space, float(np.max(np.abs(ctrl))))


def test_fit_with_no_node_in_a_direction_raises() -> None:
    """A tensor fit with an empty node array raises, as the previous release did.

    Oracle: the previous release's error (NumPy's reshape of the empty value tensor). A port that
    solved the empty system returned an all-zero spline instead.
    """
    space = create_uniform_space(3, 4)
    with pytest.raises(ValueError, match="cannot reshape array of size 0"):
        fit_bspline(np.array([]), [np.array([])], space)
