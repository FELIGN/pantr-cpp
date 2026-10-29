"""The benchmark cases: each a name, the stage that owns it, a size, and a setup.

A case's setup runs once per process, outside the timing, and returns the zero-argument callable
that is timed. Cases call pantr's public API, so the same case measures ``main`` and the port
branch alike (design/port_design.md, section 9.1; each stage adds the cases of what it ports).

Sizes: ``tiny`` is one point or one object (the per-call overhead regime), ``medium`` 1e3 items,
``large`` 1e6 points for the 1D tabulations and a 512 x 512 lattice for the ND ones.

The ``r6`` group is the stage-0 stable-ABI check (section 9.4): the per-call cost of the compiled
binding on ``tiny`` inputs. On a tree with the extension it times ``pantr._pantr._testing``'s
probes through one-line Python wrappers, the shape of a ported function; on ``main``, which has no
extension, the same case times the public operation of that shape (named in each case's ``what``).
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
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


def _has_extension() -> bool:
    return importlib.util.find_spec("pantr._ext") is not None


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
                lambda n=n: (lambda: quad.get_gauss_legendre_1d(n)),
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

    def first_call(fn: Callable[[], object], caches: list[Any]) -> Callable[[], object]:
        def call() -> object:
            for c in caches:
                c.cache_clear()
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
            lambda: first_call(
                lambda: cb.compute_lagrange_to_bernstein_1d(5),
                [cb._cached_lagrange_to_bernstein_matrix],
            ),
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
                lambda fn=fn: (lambda: fn(np.float64)),
            )
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
    if _has_extension():
        from pantr._ext import ext  # noqa: PLC0415

        probes = ext._testing

        def scalar(x: float) -> float:
            return probes.probe_scalar(x)

        def array(degree: int, pts: np.ndarray) -> np.ndarray:
            return probes.probe_bernstein(degree, pts)

        class Wrapped:
            def __init__(self) -> None:
                self._h = probes.ProbeObject(3)

            @property
            def degree(self) -> int:
                return self._h.degree

        obj: Any = Wrapped()
        return [
            Case(
                "r6.scalar.tiny",
                "r6",
                "tiny",
                "probe_scalar(0.5) through a wrapper",
                lambda: (lambda: scalar(0.5)),
            ),
            Case(
                "r6.one_point.tiny",
                "r6",
                "tiny",
                "probe_bernstein(3, 1 point) through a wrapper",
                lambda: (lambda: array(3, point)),
            ),
            Case(
                "r6.tiny_array.tiny",
                "r6",
                "tiny",
                "probe_bernstein(3, 8 points) through a wrapper",
                lambda: (lambda: array(3, eight)),
            ),
            Case(
                "r6.property.tiny",
                "r6",
                "tiny",
                "wrapper property over a bound property",
                lambda: (lambda: obj.degree),
            ),
        ]
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


def all_cases() -> list[Case]:
    """Return every case this tree can run.

    Returns:
        list[Case]: The S1 cases, then the r6 cases.
    """
    return _s1_cases() + _r6_cases()
