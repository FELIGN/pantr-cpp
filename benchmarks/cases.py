"""The benchmark cases: each a name, the stage that owns it, a size, and a setup.

A case's setup runs once per process, outside the timing, and returns the zero-argument callable
that is timed. Cases call pantr's public API, so the same case measures ``main`` and the port
branch alike (design/port_design.md, section 9.1; each stage adds the cases of what it ports).

Sizes: ``tiny`` is one point or one object (the per-call overhead regime), ``medium`` 1e3 items,
``large`` 1e6 points for the 1D tabulations and a 512 x 512 lattice for the ND ones; the B-spline
tabulations at scattered ND points take 1e5 points as ``large`` (a 3D degree-3 result is 64 values
per point).

The ``r6`` group is the stage-0 stable-ABI check (section 9.4): the per-call cost of the compiled
binding on ``tiny`` inputs. Since the quadrature/basis stage it times the same public operations on
every tree (each case's ``what``), which are wrappers over the compiled module on the port branch;
the stage-0 measurement timed one-line wrappers over probes of the module that no longer exist.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np

SIZES = {"tiny": 1, "medium": 1_000, "large": 1_000_000}
ND_SIDE = {"tiny": 1, "medium": 32, "large": 512}


@dataclass(frozen=True)
class Case:
    """One benchmark case.

    Attributes:
        name (str): Unique name, ``<group>.<operation>.<size>``.
        stage (str): The port stage that owns the operation (``S1``...), or ``r6``.
        size (str): ``tiny``, ``medium`` or ``large``.
        what (str): What is timed, for the reader of a result table.
        setup (Callable[[], Callable[[], Any]]): Builds the timed callable.
    """

    name: str
    stage: str
    size: str
    what: str
    setup: Callable[[], Callable[[], Any]]


def _points(n: int) -> np.ndarray:
    rng = np.random.default_rng(1234)
    return rng.random(n)


def _returning(fn: Callable[[], Any]) -> Callable[[], Any]:
    """Return ``fn`` itself: a setup that has nothing to prepare."""
    return fn


def _s1_cases() -> list[Case]:
    from pantr import (  # noqa: PLC0415
        basis,
        change_basis,
        quad,
        tolerance,
    )
    from pantr.basis import LagrangeVariant  # noqa: PLC0415

    cases: list[Case] = []
    degree = 3
    tab_1d: dict[str, Callable[[np.ndarray], Any]] = {
        "tabulate_bernstein_1d": lambda x: basis.tabulate_bernstein_1d(degree, x),
        "tabulate_cardinal_bspline_1d": lambda x: basis.tabulate_cardinal_bspline_1d(degree, x),
        "tabulate_legendre_1d": lambda x: basis.tabulate_legendre_1d(degree, x),
        "tabulate_lagrange_1d": lambda x: basis.tabulate_lagrange_1d(
            degree, LagrangeVariant.EQUISPACES, x
        ),
    }
    for op, f in tab_1d.items():
        for size, n in SIZES.items():

            def setup(f: Callable[[np.ndarray], Any] = f, n: int = n) -> Callable[[], Any]:
                x = _points(n)
                return lambda: f(x)

            cases.append(Case(f"S1.{op}.{size}", "S1", size, f"{op}(3, {n} points)", setup))

    tab_nd: dict[str, Callable[[Any], Any]] = {
        "tabulate_bernstein_2d": lambda p: basis.tabulate_bernstein([degree, degree], p),
        "tabulate_cardinal_bspline_2d": lambda p: basis.tabulate_cardinal_bspline(
            [degree, degree], p
        ),
    }
    for op, f in tab_nd.items():
        for size, side in ND_SIDE.items():

            def setup_nd(f: Callable[[Any], Any] = f, side: int = side) -> Callable[[], Any]:
                lattice = quad.PointsLattice([_points(side), _points(side)])
                return lambda: f(lattice)

            cases.append(
                Case(
                    f"S1.{op}.{size}", "S1", size, f"{op}([3, 3], {side}x{side} lattice)", setup_nd
                )
            )

    for n in (8, 64):
        cases.append(
            Case(
                f"S1.get_gauss_legendre_1d.n{n}",
                "S1",
                "tiny",
                f"get_gauss_legendre_1d({n})",
                partial(_returning, partial(quad.get_gauss_legendre_1d, n)),
            )
        )
    cases.append(
        Case(
            "S1.get_tanh_sinh_1d.n64",
            "S1",
            "tiny",
            "get_tanh_sinh_1d(64)",
            lambda: (lambda: quad.get_tanh_sinh_1d(64)),
        )
    )

    def first_call(fn: Callable[[], object]) -> Callable[[], object]:
        # Empty the change-of-basis caches before each call: main's functools caches, or the C++
        # process tables on the port branch. The public builder timed is uncached on both, so
        # this pins that no cache an earlier call filled can serve it.
        if importlib.util.find_spec("pantr._testing") is not None:
            from pantr import _testing  # noqa: PLC0415

            clear: Callable[[], object] = _testing.change_basis_cache_clear
        else:
            # main's lru_cache wrapper (its attribute is unknown to this tree's types).
            clear = getattr(cb._cached_lagrange_to_bernstein_matrix, "cache_clear")  # noqa: B009

        def call() -> object:
            clear()
            return fn()

        return call

    cb = change_basis
    cases.append(
        Case(
            "S1.compute_lagrange_to_bernstein_1d.repeated",
            "S1",
            "tiny",
            "compute_lagrange_to_bernstein_1d(5), cached",
            lambda: (lambda: cb.compute_lagrange_to_bernstein_1d(5)),
        )
    )
    cases.append(
        Case(
            "S1.compute_lagrange_to_bernstein_1d.first",
            "S1",
            "tiny",
            "compute_lagrange_to_bernstein_1d(5), cache cleared before each call",
            lambda: first_call(lambda: cb.compute_lagrange_to_bernstein_1d(5)),
        )
    )
    cases.append(
        Case(
            "S1.compute_bernstein_to_cardinal_1d.repeated",
            "S1",
            "tiny",
            "compute_bernstein_to_cardinal_1d(5), cached",
            lambda: (lambda: cb.compute_bernstein_to_cardinal_1d(5)),
        )
    )
    for name in ("get_default", "get_strict", "get_conservative", "get_machine_epsilon"):
        fn = getattr(tolerance, name)
        cases.append(
            Case(
                f"S1.tolerance.{name}",
                "S1",
                "tiny",
                f"tolerance.{name}(np.float64)",
                partial(_returning, partial(fn, np.float64)),
            )
        )
    return cases


def _s2_geometry_cases() -> list[Case]:
    """The AABB and AffineTransform cases of the S2 stage (C2.1)."""
    from pantr.geometry import AABB  # noqa: PLC0415
    from pantr.transform import AffineTransform  # noqa: PLC0415

    cases: list[Case] = []

    def add(name: str, size: str, what: str, setup: Callable[[], Callable[[], Any]]) -> None:
        cases.append(Case(f"S2.{name}.{size}", "S2", size, what, setup))

    box = AABB([0.0, 0.0, 0.0], [1.0, 2.0, 3.0])
    other = AABB([0.5, -1.0, 1.0], [4.0, 1.0, 2.0])
    rot = AffineTransform.rotation_3d(0.3, [1.0, 2.0, 2.0], center=[0.5, 0.5, 0.5])
    x = np.array([0.5, 1.0, 1.5])
    add("aabb_construct", "tiny", "AABB(lo, hi), 3-D", lambda: (lambda: AABB([0, 0, 0], [1, 2, 3])))
    add("aabb_overlaps", "tiny", "AABB.overlaps, 3-D", lambda: (lambda: box.overlaps(other)))
    add("aabb_union", "tiny", "AABB.union, 3-D", lambda: (lambda: box.union(other)))
    add("aabb_intersect", "tiny", "AABB.intersect, 3-D", lambda: (lambda: box.intersect(other)))
    add(
        "aabb_contains_point",
        "tiny",
        "AABB.contains_point, 3-D",
        lambda: (lambda: box.contains_point(x)),
    )
    add("aabb_pad", "tiny", "AABB.pad(0.1), 3-D", lambda: (lambda: box.pad(0.1)))
    add(
        "aabb_transform",
        "tiny",
        "AABB.transform(rotation), 3-D",
        lambda: (lambda: box.transform(rot)),
    )
    add("affine_compose", "tiny", "AffineTransform @, 3-D", lambda: (lambda: rot @ rot))
    for size, n in SIZES.items():

        def setup_apply(n: int = n) -> Callable[[], Any]:
            pts = np.random.default_rng(7).random((n, 3))
            return lambda: rot(pts)

        add("affine_apply", size, f"AffineTransform.__call__, {n} 3-D points", setup_apply)

    return cases


def _s2_bezier_cases() -> list[Case]:
    """The Bezier value-type cases of the S2 stage (C2.2)."""
    from pantr.bezier import Bezier  # noqa: PLC0415
    from pantr.quad import PointsLattice  # noqa: PLC0415
    from pantr.transform import AffineTransform  # noqa: PLC0415

    cases: list[Case] = []

    def add(name: str, size: str, what: str, setup: Callable[[], Callable[[], Any]]) -> None:
        cases.append(Case(f"S2.{name}.{size}", "S2", size, what, setup))

    rot = AffineTransform.rotation_3d(0.3, [1.0, 2.0, 2.0], center=[0.5, 0.5, 0.5])
    rng = np.random.default_rng(11)
    curve_cp = rng.random((4, 3))
    surf_cp = rng.random((4, 4, 3))
    curve = Bezier(curve_cp)
    surf = Bezier(surf_cp)
    rational = Bezier(
        np.concatenate([curve_cp, rng.random((4, 1)) + 0.5], axis=1), is_rational=True
    )
    add(
        "bezier_construct",
        "tiny",
        "Bezier(cp), degree 3, rank 3",
        lambda: (lambda: Bezier(curve_cp)),
    )
    add(
        "bezier_control_points",
        "tiny",
        "Bezier.control_points",
        lambda: (lambda: curve.control_points),
    )
    for size, n in SIZES.items():

        def setup_curve(n: int = n) -> Callable[[], Any]:
            t = _points(n)
            return lambda: curve.evaluate(t)

        def setup_rational(n: int = n) -> Callable[[], Any]:
            t = _points(n)
            return lambda: rational.evaluate(t)

        def setup_curve_d1(n: int = n) -> Callable[[], Any]:
            t = _points(n)
            return lambda: curve.evaluate_derivatives(t, 1)

        def setup_surf(n: int = n) -> Callable[[], Any]:
            uv = np.random.default_rng(3).random((n, 2))
            return lambda: surf.evaluate(uv)

        def setup_surf_d(n: int = n) -> Callable[[], Any]:
            uv = np.random.default_rng(3).random((n, 2))
            return lambda: surf.evaluate_derivatives(uv, [1, 1])

        add("bezier_evaluate_curve", size, f"evaluate, degree-3 curve, {n} points", setup_curve)
        add(
            "bezier_evaluate_rational",
            size,
            f"evaluate, rational degree-3 curve, {n} points",
            setup_rational,
        )
        add(
            "bezier_derivatives_curve",
            size,
            f"evaluate_derivatives(1), degree-3 curve, {n} points",
            setup_curve_d1,
        )
        add(
            "bezier_evaluate_surface",
            size,
            f"evaluate, degree-(3, 3) surface, {n} points",
            setup_surf,
        )
        add(
            "bezier_derivatives_surface",
            size,
            f"evaluate_derivatives([1, 1]), surface, {n} points",
            setup_surf_d,
        )
    for size, side in ND_SIDE.items():

        def setup_lattice(side: int = side) -> Callable[[], Any]:
            lattice = PointsLattice([_points(side), _points(side)])
            return lambda: surf.evaluate(lattice)

        add(
            "bezier_evaluate_lattice",
            size,
            f"evaluate, surface, {side}x{side} lattice",
            setup_lattice,
        )
    big = Bezier(rng.random((33, 33, 3)))
    add(
        "bezier_split",
        "tiny",
        "split(0, 0.3), degree-(3, 3) surface",
        lambda: (lambda: surf.split(0, 0.3)),
    )
    add(
        "bezier_split",
        "medium",
        "split(0, 0.3), degree-(32, 32) surface",
        lambda: (lambda: big.split(0, 0.3)),
    )
    add(
        "bezier_slice",
        "tiny",
        "slice(1, 0.3), degree-(3, 3) surface",
        lambda: (lambda: surf.slice(1, 0.3)),
    )
    add(
        "bezier_restrict",
        "tiny",
        "restrict((0.2, 0.7)), degree-3 curve",
        lambda: (lambda: curve.restrict((0.2, 0.7))),
    )
    add(
        "bezier_elevate_degree",
        "tiny",
        "elevate_degree(2), degree-3 curve",
        lambda: (lambda: curve.elevate_degree(2)),
    )
    add(
        "bezier_elevate_degree",
        "medium",
        "elevate_degree(2), degree-(32, 32) surface",
        lambda: (lambda: big.elevate_degree(2)),
    )
    add(
        "bezier_derivative",
        "tiny",
        "derivative(0), degree-(3, 3) surface",
        lambda: (lambda: surf.derivative(0)),
    )
    add(
        "bezier_multiply",
        "tiny",
        "multiply, degree-(3, 3) surfaces",
        lambda: (lambda: surf.multiply(surf)),
    )
    add(
        "bezier_reverse",
        "tiny",
        "reverse(0), degree-(3, 3) surface",
        lambda: (lambda: surf.reverse(0)),
    )
    add(
        "bezier_transform",
        "tiny",
        "transform(rotation), degree-(3, 3) surface",
        lambda: (lambda: surf.transform(rot)),
    )
    return cases


def _s2_approx_roots_cases() -> list[Case]:
    """The degree-reduction, fitting and root-finding cases of the S2 stage (C2.3, C2.4)."""
    from pantr.bezier import (  # noqa: PLC0415
        Bezier,
        find_monotone_root,
        find_roots,
        fit_bezier,
        interpolate_bezier,
    )

    cases: list[Case] = []

    def add(name: str, size: str, what: str, setup: Callable[[], Callable[[], Any]]) -> None:
        cases.append(Case(f"S2.{name}.{size}", "S2", size, what, setup))

    rng = np.random.default_rng(13)
    curve = Bezier(rng.random((7, 3)))
    surf = Bezier(rng.random((13, 13, 3)))
    # Degree-3 curve elevated to 9 and 20: minimize_degree recovers degree 3 in each.
    base = Bezier(rng.random((4, 3)))
    elevated = base.elevate_degree(6)
    elevated_big = base.elevate_degree(17)
    add(
        "bezier_reduce_degree",
        "tiny",
        "reduce_degree(2), degree-6 curve",
        lambda: (lambda: curve.reduce_degree(2)),
    )
    add(
        "bezier_reduce_degree",
        "medium",
        "reduce_degree(3), degree-(12, 12) surface",
        lambda: (lambda: surf.reduce_degree(3)),
    )
    add(
        "bezier_minimize_degree",
        "tiny",
        "minimize_degree, degree-9 curve of degree 3",
        lambda: (lambda: elevated.minimize_degree()),
    )
    add(
        "bezier_minimize_degree",
        "medium",
        "minimize_degree, degree-20 curve of degree 3",
        lambda: (lambda: elevated_big.minimize_degree()),
    )

    def setup_interp(dims: list[int]) -> Callable[[], Any]:
        def func(lattice: Any) -> Any:  # noqa: ANN401
            x = lattice.get_all_points()
            return np.sin(3.0 * x[:, 0]) * np.cos(2.0 * x[:, -1])

        return lambda: interpolate_bezier(func, dims)

    add("interpolate_bezier", "tiny", "interpolate_bezier, 8 nodes", lambda: setup_interp([8]))
    add(
        "interpolate_bezier",
        "medium",
        "interpolate_bezier, 32x32 nodes",
        lambda: setup_interp([32, 32]),
    )

    def setup_fit_tensor(n: int) -> Callable[[], Any]:
        nodes = [np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, n)]
        values = np.random.default_rng(5).random((n, n, 2))
        return lambda: fit_bezier(values, nodes, degree=n // 2)

    def setup_fit_scattered(n: int) -> Callable[[], Any]:
        pts = np.random.default_rng(6).random((n, 2))
        values = np.random.default_rng(7).random((n, 2))
        return lambda: fit_bezier(values, pts, degree=[3, 3])

    add("fit_bezier_tensor", "tiny", "fit_bezier, 6x6 grid, degree 3", lambda: setup_fit_tensor(6))
    add(
        "fit_bezier_tensor",
        "medium",
        "fit_bezier, 32x32 grid, degree 16",
        lambda: setup_fit_tensor(32),
    )
    add(
        "fit_bezier_scattered",
        "tiny",
        "fit_bezier, 20 scattered points, degree (3, 3)",
        lambda: setup_fit_scattered(20),
    )
    add(
        "fit_bezier_scattered",
        "medium",
        "fit_bezier, 1000 scattered points, degree (3, 3)",
        lambda: setup_fit_scattered(1000),
    )

    roots3 = Bezier(np.array([1.0, -2.0, 1.5, -0.5]))
    roots12 = Bezier(np.cos(np.linspace(0.0, 7.0, 13)))
    add("find_roots", "tiny", "find_roots, degree 3 (Yuksel)", lambda: (lambda: find_roots(roots3)))
    add(
        "find_roots_degree12",
        "tiny",
        "find_roots, degree 12 (clipping)",
        lambda: (lambda: find_roots(roots12)),
    )

    def setup_batch(n: int) -> Callable[[], Any]:
        coeffs = np.random.default_rng(8).standard_normal((n, 6))
        batch = [Bezier(c) for c in coeffs]
        return lambda: find_roots(batch)

    add(
        "find_roots_batch", "medium", "find_roots, batch of 1e3 degree-5", lambda: setup_batch(1000)
    )
    add(
        "find_roots_batch",
        "large",
        "find_roots, batch of 1e4 degree-5",
        lambda: setup_batch(10_000),
    )
    mono = Bezier(np.array([-1.0, -0.4, 0.3, 0.8, 1.2]))
    add(
        "find_monotone_root",
        "tiny",
        "find_monotone_root, degree 4",
        lambda: (lambda: find_monotone_root(mono)),
    )

    def setup_mono_batch(n: int) -> Callable[[], Any]:
        coeffs = np.sort(np.random.default_rng(9).standard_normal((n, 5)), axis=1)
        batch = [Bezier(c) for c in coeffs]
        return lambda: find_monotone_root(batch)

    add(
        "find_monotone_root_batch",
        "large",
        "find_monotone_root, batch of 1e4 degree-4",
        lambda: setup_mono_batch(10_000),
    )
    return cases


def _r6_cases() -> list[Case]:
    from pantr import (  # noqa: PLC0415
        basis,
        quad,
        tolerance,
    )

    point = np.array([0.3])
    eight = _points(8)
    lattice = quad.PointsLattice([eight, eight])
    return [
        Case(
            "r6.scalar.tiny",
            "r6",
            "tiny",
            "tolerance.get_default(np.float64)",
            lambda: (lambda: tolerance.get_default(np.float64)),
        ),
        Case(
            "r6.one_point.tiny",
            "r6",
            "tiny",
            "tabulate_bernstein_1d(3, 1 point)",
            lambda: (lambda: basis.tabulate_bernstein_1d(3, point)),
        ),
        Case(
            "r6.tiny_array.tiny",
            "r6",
            "tiny",
            "tabulate_bernstein_1d(3, 8 points)",
            lambda: (lambda: basis.tabulate_bernstein_1d(3, eight)),
        ),
        Case("r6.property.tiny", "r6", "tiny", "PointsLattice.dim", lambda: (lambda: lattice.dim)),
    ]


def _s4_space_cases() -> list[Case]:
    """The B-spline space cases of the S4 stage (C4.1): construction and basis tabulation."""
    from pantr.bspline import BsplineSpace, BsplineSpace1D  # noqa: PLC0415
    from pantr.quad import PointsLattice  # noqa: PLC0415

    cases: list[Case] = []
    degree = 3

    def open_knots(num_intervals: int) -> np.ndarray:
        inner = np.linspace(0.0, 1.0, num_intervals + 1)
        return np.concatenate([np.zeros(degree), inner, np.ones(degree)])

    for size, n in SIZES.items():

        def setup_construct(n: int = n) -> Callable[[], Any]:
            knots = open_knots(n)
            return lambda: BsplineSpace1D(knots, degree)

        cases.append(
            Case(
                f"S4.bspline_space_1d_construct.{size}",
                "S4",
                size,
                f"BsplineSpace1D(open knots of {n} intervals, {degree})",
                setup_construct,
            )
        )

    # Ten intervals, so the general (Cox-de Boor) kernel runs, not the Bezier-like one.
    space_1d = BsplineSpace1D(open_knots(10), degree)
    for size, n in SIZES.items():

        def setup_1d(n: int = n) -> Callable[[], Any]:
            x = _points(n)
            return lambda: space_1d.tabulate_basis(x)

        cases.append(
            Case(
                f"S4.bspline_tabulate_basis_1d.{size}",
                "S4",
                size,
                f"BsplineSpace1D.tabulate_basis({n} points), degree {degree}, 10 intervals",
                setup_1d,
            )
        )

    nd_points = {"tiny": 1, "medium": 1_000, "large": 100_000}
    for dim in (2, 3):
        space = BsplineSpace([space_1d] * dim)
        for size, n in nd_points.items():

            def setup_nd(
                n: int = n, dim: int = dim, space: BsplineSpace = space
            ) -> Callable[[], Any]:
                pts = np.random.default_rng(1234).random((n, dim))
                return lambda: space.tabulate_basis(pts)

            cases.append(
                Case(
                    f"S4.bspline_tabulate_basis_{dim}d.{size}",
                    "S4",
                    size,
                    f"BsplineSpace.tabulate_basis({n} x {dim} points), degree {degree}",
                    setup_nd,
                )
            )

    space_2d = BsplineSpace([space_1d, space_1d])
    for size, side in ND_SIDE.items():

        def setup_lattice(side: int = side) -> Callable[[], Any]:
            lattice = PointsLattice([_points(side), _points(side)])
            return lambda: space_2d.tabulate_basis(lattice)

        cases.append(
            Case(
                f"S4.bspline_tabulate_basis_2d_lattice.{size}",
                "S4",
                size,
                f"BsplineSpace.tabulate_basis({side}x{side} lattice), degree {degree}",
                setup_lattice,
            )
        )
    return cases


def _s4_bspline_cases() -> list[Case]:  # noqa: PLR0915
    """The B-spline cases of the S4 stage (C4.2): the `Bspline` value and its operations.

    Operation sizes count intervals (or inserted/removed knots) rather than points: ``medium`` is
    1e3 and ``large`` 1e5. Two operations take smaller ``large`` sizes because main's cost grows
    faster than linearly: knot removal 1e4 (each removal is a sweep), and the conversion to periodic
    form 1e3 (a dense QR factorization of the Oslo matrix).
    """
    from pantr.bspline import (  # noqa: PLC0415
        Bspline,
        BsplineSpace,
        BsplineSpace1D,
        create_uniform_open_knots,
        create_uniform_periodic_knots,
        find_roots,
    )
    from pantr.quad import PointsLattice  # noqa: PLC0415

    cases: list[Case] = []
    degree = 3
    rank = 3
    op_sizes = {"tiny": 1, "medium": 1_000, "large": 100_000}
    nd_points = {"tiny": 1, "medium": 1_000, "large": 100_000}

    def curve(num_intervals: int) -> Bspline:
        knots = create_uniform_open_knots(num_intervals, degree)
        space = BsplineSpace([BsplineSpace1D(knots, degree)])
        cp = np.random.default_rng(1234).random((num_intervals + degree, rank))
        return Bspline(space, cp)

    def add(name: str, size: str, what: str, setup: Callable[[], Callable[[], Any]]) -> None:
        cases.append(Case(f"S4.{name}.{size}", "S4", size, what, setup))

    for size, n in op_sizes.items():

        def setup_construct(n: int = n) -> Callable[[], Any]:
            space = curve(n).space
            cp = np.random.default_rng(1234).random((n + degree, rank))
            return lambda: Bspline(space, cp)

        add("bspline_construct", size, f"Bspline(open space, {n} intervals)", setup_construct)

    ten = curve(10)
    for size, n in SIZES.items():

        def setup_eval_1d(n: int = n) -> Callable[[], Any]:
            x = _points(n)
            return lambda: ten.evaluate(x)

        add("bspline_evaluate_1d", size, f"Bspline.evaluate({n} points), 1D", setup_eval_1d)

    space_1d = ten.space.spaces[0]
    for dim in (2, 3):
        nd_space = BsplineSpace([space_1d] * dim)
        nd = Bspline(nd_space, np.random.default_rng(1234).random((*nd_space.num_basis, 1)))
        for size, n in nd_points.items():

            def setup_eval_nd(n: int = n, dim: int = dim, nd: Bspline = nd) -> Callable[[], Any]:
                pts = np.random.default_rng(1234).random((n, dim))
                return lambda: nd.evaluate(pts)

            add(f"bspline_evaluate_{dim}d", size, f"evaluate({n} x {dim} points)", setup_eval_nd)

    surface_space = BsplineSpace([space_1d] * 2)
    surface = Bspline(
        surface_space, np.random.default_rng(1234).random((*surface_space.num_basis, 1))
    )
    for size, side in ND_SIDE.items():

        def setup_lattice(side: int = side) -> Callable[[], Any]:
            lattice = PointsLattice([_points(side), _points(side)])
            return lambda: surface.evaluate(lattice)

        add("bspline_evaluate_2d_lattice", size, f"evaluate({side}x{side} lattice)", setup_lattice)

    volume_space = BsplineSpace([space_1d] * 3)
    volume = Bspline(volume_space, np.random.default_rng(1234).random((*volume_space.num_basis, 1)))
    for size, side in {"tiny": 1, "medium": 10, "large": 64}.items():

        def setup_lattice_3d(side: int = side) -> Callable[[], Any]:
            lattice = PointsLattice([_points(side)] * 3)
            return lambda: volume.evaluate(lattice)

        add("bspline_evaluate_3d_lattice", size, f"evaluate({side}^3 lattice)", setup_lattice_3d)

    for size, n in nd_points.items():

        def setup_derivs(n: int = n) -> Callable[[], Any]:
            pts = np.random.default_rng(1234).random((n, 2))
            return lambda: surface.evaluate_derivatives(pts, [1, 1])

        add("bspline_evaluate_derivatives_2d", size, f"d2/dudv at {n} points", setup_derivs)

    for size, n in op_sizes.items():

        def setup_insert(n: int = n) -> Callable[[], Any]:
            knots = (np.arange(n) + 0.5) / n
            return lambda: ten.insert_knots(knots)

        add("bspline_insert_knots", size, f"insert {n} knots, 10-interval cubic", setup_insert)

        def setup_remove(n: int = min(n, 10_000)) -> Callable[[], Any]:
            knots = (np.arange(n) + 0.5) / n
            refined = curve(1).insert_knots(knots)
            return lambda: refined.remove_knots(knots)

        add("bspline_remove_knots", size, f"remove {min(n, 10_000)} knots, cubic", setup_remove)

        def setup_elevate(n: int = n) -> Callable[[], Any]:
            c = curve(n)
            return lambda: c.elevate_degree(1)

        add("bspline_elevate_degree", size, f"elevate a {n}-interval cubic by 1", setup_elevate)

        def setup_to_beziers(n: int = n) -> Callable[[], Any]:
            c = curve(n)
            # `to_beziers` caches its result on the instance, so time it on a fresh value.
            return lambda: Bspline(c.space, c.control_points).to_beziers()

        add("bspline_to_beziers", size, f"to_beziers of a {n}-interval cubic", setup_to_beziers)

        def setup_to_periodic(n: int = n) -> Callable[[], Any]:
            m = min(max(n, degree + 1), 1_000)
            knots = create_uniform_periodic_knots(m, degree)
            space = BsplineSpace([BsplineSpace1D(knots, degree, periodic=True)])
            periodic = Bspline(space, np.random.default_rng(1234).random((m, rank)))
            open_curve = periodic.to_open_bspline()
            return lambda: open_curve.to_periodic()

        add(
            "bspline_to_periodic",
            size,
            f"to_periodic, open cubic of {min(max(n, degree + 1), 1_000)} intervals",
            setup_to_periodic,
        )

    # Root finding on main grows faster than linearly in the number of intervals (each reported
    # root shifts the rest of the working window down): `large` is 4e3 intervals.
    for size, n in {"tiny": 1, "medium": 1_000, "large": 4_000}.items():

        def setup_find_roots(n: int = n) -> Callable[[], Any]:
            knots = create_uniform_open_knots(n, degree)
            space = BsplineSpace([BsplineSpace1D(knots, degree)])
            coeffs = np.random.default_rng(1234).uniform(-1.0, 1.0, (n + degree, 1))
            scalar = Bspline(space, coeffs)
            return lambda: find_roots(scalar)

        add(
            "bspline_find_roots",
            size,
            f"find_roots, random cubic of {n} intervals",
            setup_find_roots,
        )

    # The product re-expresses its Bezier form on the minimal knot vector by a dense least-squares
    # solve, whose cost is cubic in the number of intervals: `large` is 100 intervals.
    for size, n in {"tiny": 1, "medium": 10, "large": 100}.items():

        def setup_multiply(n: int = n) -> Callable[[], Any]:
            c = curve(n)
            return lambda: c.multiply(c)

        add("bspline_multiply", size, f"multiply two {n}-interval cubics", setup_multiply)
    return cases


def _s4_extraction_cases() -> list[Case]:
    """The spanwise element extraction cases of the S4 stage (C5.2): construction and application.

    A cubic space of ``n`` uniform intervals per direction in three directions: ``tiny`` is 2 per
    direction, ``medium`` 10 (1e3 elements), ``large`` 46 (97 336 elements, the largest a batch of
    the shape below reasonably is). The per-cell ``apply`` is timed as a Python loop over cells:
    one call on ``tiny``, then 1e3 and, on ``large``, 1e4 calls (a loop over every cell of
    ``large`` would make a repeat too long). The batch ``apply_many`` runs every element of the
    space.
    """
    from pantr.bspline import (  # noqa: PLC0415
        BsplineSpace,
        BsplineSpace1D,
        SpanwiseElementExtraction,
        create_uniform_open_knots,
    )

    degree = 3
    side = {"tiny": 2, "medium": 10, "large": 46}
    loop = {"tiny": 1, "medium": 1_000, "large": 10_000}
    cases: list[Case] = []

    def cubic_space(n: int) -> BsplineSpace:
        knots = create_uniform_open_knots(n, degree)
        return BsplineSpace([BsplineSpace1D(knots, degree)] * 3)

    for size, n in side.items():

        def setup_construct(n: int = n) -> Callable[[], Any]:
            space = cubic_space(n)
            return lambda: SpanwiseElementExtraction(space, "bezier")

        cases.append(
            Case(
                f"S4.spanwise_extraction_construct.{size}",
                "S4",
                size,
                f"SpanwiseElementExtraction(cubic space, {n} intervals per direction, 'bezier')",
                setup_construct,
            )
        )

    for size, calls in loop.items():

        def setup_apply(calls: int = calls, n: int = side[size]) -> Callable[[], Any]:
            ext = SpanwiseElementExtraction(cubic_space(n), "bezier")
            n_in = int(np.prod(ext.input_shape_per_dir))
            v = np.random.default_rng(1234).random(n_in)
            total = ext.num_total_intervals
            cells = [int(c) for c in np.linspace(0, total - 1, calls).astype(np.intp)]

            def run() -> object:
                out = None
                for cell in cells:
                    out = ext.apply(v, cell)
                return out

            return run

        cases.append(
            Case(
                f"S4.spanwise_extraction_apply.{size}",
                "S4",
                size,
                f"{calls} calls of apply on a cubic 3D extraction, {side[size]}^3 elements",
                setup_apply,
            )
        )

    for size, n in side.items():

        def setup_apply_many(n: int = n) -> Callable[[], Any]:
            ext = SpanwiseElementExtraction(cubic_space(n), "bezier")
            n_in = int(np.prod(ext.input_shape_per_dir))
            cells = np.arange(ext.num_total_intervals)
            v = np.random.default_rng(1234).random((cells.size, n_in))
            return lambda: ext.apply_many(v, cells)

        cases.append(
            Case(
                f"S4.spanwise_extraction_apply_many.{size}",
                "S4",
                size,
                f"apply_many over all {n}^3 elements of a cubic 3D extraction",
                setup_apply_many,
            )
        )
    return cases


def all_cases() -> list[Case]:
    """Return every case this tree can run.

    Returns:
        list[Case]: The S1 cases, the S2 cases, the S4 cases, then the r6 cases.
    """
    return (
        _s1_cases()
        + _s2_geometry_cases()
        + _s2_bezier_cases()
        + _s2_approx_roots_cases()
        + _s4_space_cases()
        + _s4_bspline_cases()
        + _s4_extraction_cases()
        + _r6_cases()
    )
