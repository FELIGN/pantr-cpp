"""What the C++ port of the B-spline spaces changed or must have kept, pinned from Python.

Three groups, each answering a way the port could differ from the previous release without any
suite noticing:

* A zero divisor in the derivative recurrence raises ``ZeroDivisionError`` at every batch size.
  The previous release raised it below 4096 points only (the serial kernel); past that the
  parallel kernel divided in IEEE arithmetic and returned ``inf`` or ``nan``. The test points sit
  at 1, 4095, 4096, 5000 and 30000, with the offending point first, in the middle and last.
* The knot factories and ``get_greville_abscissae`` compute the values the previous release
  computed. The oracle is NumPy (``linspace``, ``mean``, ``mod``) assembled in the test from the
  documented formula, and the comparison is within a tolerance derived from the number of rounded
  operations (never bit identity, which NumPy versions do not promise), plus properties that need
  no NumPy: the structure of the multiplicities, the uniform spacing, the sizes, and the integers
  of a cardinal vector.
* Pickling and copying: a space round-trips by value (knots, tolerance, snapped or not), and a
  pickle written by the previous release (a plain object with ``_knots``, ``_degree``,
  ``_periodic`` and ``_tol``) still loads, with its stored tolerance.
"""

from __future__ import annotations

import copy
import functools
import pickle
import sys
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from pantr.bspline import (
    BsplineSpace,
    BsplineSpace1D,
    create_cardinal_knots,
    create_uniform_open_knots,
    create_uniform_periodic_knots,
    get_greville_abscissae,
)

_Array = npt.NDArray[np.float32 | np.float64]

_DTYPES = [np.float32, np.float64]
_DTYPE_IDS = ["f32", "f64"]


def _eps(dtype: type) -> float:
    """Return the machine epsilon of ``dtype`` as a Python float."""
    return float(np.finfo(dtype).eps)


# ---------------------------------------------------------------------------
# A zero divisor in the derivative recurrence
# ---------------------------------------------------------------------------

_LEFT_HEAVY = [0, 0, 0, 0, 1, 1, 1]
"""Degree 2 with the left end repeated four times (p + 2): the span just left of 0 is empty."""

_BAD_POINT = -1e-17
"""Left of 0 by far less than the knot tolerance (8 eps), so inside the band validation admits."""

_BATCH_SIZES = [1, 5, 4095, 4096, 5000, 30000]
"""Around the old serial/parallel threshold of 4096 points, and well past the C++ pool's."""


def _batch(n: int, dtype: type, bad_at: int | None) -> _Array:
    """Return ``n`` points of [0, 1], with ``_BAD_POINT`` at index ``bad_at`` when given."""
    pts: _Array = np.linspace(0.0, 1.0, n).astype(dtype)
    if bad_at is not None:
        pts[bad_at] = dtype(_BAD_POINT)
    return pts


@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("position", ["first", "middle", "last"])
@pytest.mark.parametrize("n", _BATCH_SIZES)
def test_a_zero_divisor_raises_at_every_batch_size(n: int, position: str, dtype: type) -> None:
    """One offending point among ``n`` raises ``ZeroDivisionError`` whatever ``n`` is.

    Catches: the status of the C++ kernel lost on the pool path (a chunk's failure not reaching
    the caller), or reported only for small batches, as the previous release did.
    """
    space = BsplineSpace1D(np.array(_LEFT_HEAVY, dtype=dtype), 2)
    at = {"first": 0, "middle": n // 2, "last": n - 1}[position]
    pts = _batch(n, dtype, at)

    with pytest.raises(ZeroDivisionError):
        space.tabulate_basis_derivatives(pts, 1, validate=False)


@pytest.mark.parametrize("n", [1, 5000])
def test_the_point_is_inside_the_tolerance_band_so_validation_does_not_stop_it(n: int) -> None:
    """With the default ``validate=True`` the same point passes validation and then raises.

    Pins the previous release's behaviour (it raised there too): the empty span is reachable by
    an input the space itself calls in-domain. Whether it should instead be evaluated on the
    first non-empty span is the owner's call; this test only records that the two agree.
    """
    space = BsplineSpace1D(_LEFT_HEAVY, 2)
    assert float(space.tolerance) > abs(_BAD_POINT)

    with pytest.raises(ZeroDivisionError):
        space.tabulate_basis_derivatives(_batch(n, np.float64, n // 2), 1)


@pytest.mark.parametrize("n", [1, 5000])
def test_order_zero_divides_by_nothing_and_stays_finite(n: int) -> None:
    """``n_deriv=0`` has no derivative row to divide, so the offending point raises nothing.

    Its values are only required to be finite and in [0, 1] (the guard of the value recurrence
    keeps them bounded); they are not pinned beyond that.
    """
    space = BsplineSpace1D(_LEFT_HEAVY, 2)

    values, first = space.tabulate_basis_derivatives(
        _batch(n, np.float64, n // 2), 0, validate=False
    )

    assert values.shape == (n, 1, 3)
    assert first.shape == (n,)
    assert np.isfinite(values).all()
    assert values.min() >= 0.0
    assert values.max() <= 1.0 + 8 * _eps(np.float64)


@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("n", [1, 4096, 5000])
def test_the_same_space_without_the_offending_point_is_the_bernstein_basis(
    n: int, dtype: type
) -> None:
    """Control: in-domain points on this space give the quadratic Bernstein basis, first index 1.

    The left function of a left end of multiplicity 4 has empty support, so on [0, 1] the other
    three are the Bernstein polynomials ((1-x)^2, 2x(1-x), x^2) and their derivatives
    (-2(1-x), 2-4x, 2x), (2, -4, 2) and 0. Tolerances: the values 16 eps (8 p eps), the first
    derivative row 16 (p + 1) eps (2p)^1 and the second 16 (p + 1) eps (2p)^2 with unit knot gaps.
    """
    space = BsplineSpace1D(np.array(_LEFT_HEAVY, dtype=dtype), 2)
    x = _batch(n, dtype, None)
    xd = x.astype(np.float64)
    want = np.empty((n, 4, 3))
    want[:, 0] = np.stack([(1 - xd) ** 2, 2 * xd * (1 - xd), xd**2], axis=1)
    want[:, 1] = np.stack([-2 * (1 - xd), 2 - 4 * xd, 2 * xd], axis=1)
    want[:, 2] = np.array([2.0, -4.0, 2.0])
    want[:, 3] = 0.0
    eps = _eps(dtype)
    tol = np.array([16 * eps, 192 * eps, 768 * eps, 3072 * eps]).reshape(1, 4, 1)

    values, first = space.tabulate_basis_derivatives(x, 3)

    assert np.all(first == 1)
    assert np.all(np.abs(values.astype(np.float64) - want) <= tol)


# ---------------------------------------------------------------------------
# The knot factories against NumPy, by the definition the previous release used
# ---------------------------------------------------------------------------

_DOMAINS = [
    pytest.param((0.0, 1.0), id="unit"),
    pytest.param((-3.0, 5.0), id="negative-start"),
    pytest.param((1000.0, 1007.0), id="offset-1e3"),
    pytest.param((-1e-3, 2e-3), id="small"),
    pytest.param((0.0, 1048576.0), id="wide-2^20"),
]


def _open_by_definition(n: int, p: int, c: int, a: Any, b: Any, dtype: type) -> _Array:  # noqa: PLR0913
    """Return the previous release's open vector.

    ``start`` and ``end`` ``p + 1`` times, the interior knots of ``np.linspace(start, end, n + 1)``
    each ``p - c`` times.
    """
    unique: _Array = np.linspace(a, b, n + 1, dtype=dtype)
    parts: list[_Array] = [np.full(p + 1, a, dtype=dtype)]
    parts += [np.full(p - c, u, dtype=dtype) for u in unique[1:-1]]
    parts.append(np.full(p + 1, b, dtype=dtype))
    return np.concatenate(parts)


def _periodic_by_definition(n: int, p: int, c: int, a: Any, b: Any, dtype: type) -> _Array:  # noqa: PLR0913
    """Return the previous release's periodic vector.

    Every ``np.linspace(start, end, n + 1)`` knot ``p - c`` times, extended by ``c + 1`` knots at
    the interval length at each end.
    """
    mult = p - c
    unique: _Array = np.linspace(a, b, n + 1, dtype=dtype)
    reach = ((b - a) / n) * (p - mult + 1)
    left: _Array = np.linspace(a - reach, a, p + 2 - mult, dtype=dtype)[:-1]
    right: _Array = np.linspace(b, b + reach, p + 2 - mult, dtype=dtype)[1:]
    return np.concatenate([left, np.repeat(unique, mult), right])


def _knot_tolerance(a: float, b: float, n: int, p: int, dtype: type) -> float:
    """Bound on the difference of two evaluations of the factory's formula, per knot.

    Each knot comes out of at most eight correctly rounded operations (``b - a``, the quotient,
    the product, the sum, twice for the extension) on quantities of size at most
    ``M = |a| + |b| + (p + 1) (b - a) / n``, so it is within ``8 u M`` of the exact value
    (``u = eps / 2``) and two evaluations differ by at most ``8 eps M``; the factor 2 is margin.
    """
    m = abs(a) + abs(b) + (p + 1) * (b - a) / n
    return 16 * _eps(dtype) * m


_GRID = [(n, p, c) for n in (1, 2, 5, 17) for p in (0, 1, 2, 3, 5) for c in range(-1, p)]


@pytest.mark.parametrize("domain", _DOMAINS)
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
def test_uniform_open_knots_match_the_numpy_definition(
    dtype: type, domain: tuple[float, float]
) -> None:
    """Open vectors over a grid of (intervals, degree, continuity) agree with ``np.linspace``.

    Catches: a changed grid formula, a wrong multiplicity, a dtype that drifts, ends that are not
    the supplied ones bit for bit.
    """
    a, b = dtype(domain[0]), dtype(domain[1])
    for n, p, c in _GRID:
        got = create_uniform_open_knots(n, p, c, domain=(a, b), dtype=dtype)
        want = _open_by_definition(n, p, c, a, b, dtype)
        tol = _knot_tolerance(float(a), float(b), n, p, dtype)
        ctx = f"n={n} p={p} c={c}"

        assert got.dtype == dtype, ctx
        assert got.shape == want.shape, ctx
        assert np.all(got[: p + 1] == a), ctx  # bit for bit
        assert np.all(got[len(got) - p - 1 :] == b), ctx
        assert np.all(np.abs(got - want) <= tol), ctx
        assert np.all(np.diff(got) >= 0), ctx
        interior = got[p + 1 : len(got) - p - 1]
        if interior.size:
            values, counts = np.unique(interior, return_counts=True)
            assert values.size == n - 1, ctx
            assert np.all(counts == p - c), ctx  # each interior knot exactly p - c times
            spacing = np.diff(np.concatenate([[a], values, [b]]).astype(np.float64))
            assert np.all(np.abs(spacing - (float(b) - float(a)) / n) <= 4 * tol), ctx


@pytest.mark.parametrize("domain", _DOMAINS)
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
def test_uniform_periodic_knots_match_the_numpy_definition(
    dtype: type, domain: tuple[float, float]
) -> None:
    """Periodic vectors agree with the definition, have the closed-form size and uniform spacing.

    The size is ``2 (c + 1) + (n + 1) (p - c)``, and the distinct knots are the grid extended by
    ``c + 1`` intervals past each end, so their spacing is one interval length throughout.
    """
    a, b = dtype(domain[0]), dtype(domain[1])
    for n, p, c in _GRID:
        got = create_uniform_periodic_knots(n, p, c, domain=(a, b), dtype=dtype)
        want = _periodic_by_definition(n, p, c, a, b, dtype)
        tol = _knot_tolerance(float(a), float(b), n, p, dtype)
        ctx = f"n={n} p={p} c={c}"

        assert got.dtype == dtype, ctx
        assert got.size == 2 * (c + 1) + (n + 1) * (p - c), ctx
        assert got.shape == want.shape, ctx
        assert np.all(np.abs(got - want) <= tol), ctx
        assert got[c + 1] == a, ctx
        assert got[got.size - c - 2] == b, ctx
        values, counts = np.unique(got, return_counts=True)
        inside = (values >= a) & (values <= b)
        assert np.all(counts[inside] == p - c), ctx  # the grid knots, each p - c times
        assert np.all(counts[~inside] == 1), ctx  # the extension knots, once
        assert (~inside).sum() == 2 * (c + 1), ctx
        length = (float(b) - float(a)) / n
        assert np.all(np.abs(np.diff(values.astype(np.float64)) - length) <= 8 * tol), ctx


@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
def test_cardinal_knots_are_the_integers_from_minus_degree(dtype: type) -> None:
    """``create_cardinal_knots(n, p)`` is ``-p, ..., n + p`` exactly (integers: no tolerance)."""
    for n in (1, 2, 7):
        for p in (0, 1, 2, 5):
            got = create_cardinal_knots(n, p, dtype)
            assert got.dtype == dtype
            np.testing.assert_array_equal(got, np.arange(-p, n + p + 1, dtype=dtype))


def test_a_knot_count_past_the_supported_size_is_a_value_error_not_an_allocation() -> None:
    """The C++ ``capacity_error`` reaches Python as ``ValueError`` before anything is allocated."""
    huge = 2**62
    with pytest.raises(ValueError, match="knot vector too large"):
        create_uniform_open_knots(huge, 3)
    with pytest.raises(ValueError, match="knot vector too large"):
        create_uniform_periodic_knots(huge, 3)
    with pytest.raises(ValueError, match="knot vector too large"):
        create_uniform_open_knots(2, huge)
    with pytest.raises(ValueError, match="knot vector too large"):
        create_cardinal_knots(huge, 2)


# ---------------------------------------------------------------------------
# Greville abscissae against np.mean / np.mod, by the previous release's definition
# ---------------------------------------------------------------------------


def _greville_by_definition(space: BsplineSpace1D) -> _Array:
    """The previous release's abscissae: ``np.mean`` of ``degree`` knots, wrapped by ``np.mod``."""
    knots, p, nb = space.knots, space.degree, space.num_basis
    if p == 0:
        g = (knots[:nb] + knots[1 : nb + 1]) / 2
    else:
        g = np.array([np.mean(knots[i + 1 : i + p + 1]) for i in range(nb)], dtype=knots.dtype)
    if space.periodic:
        a, b = space.domain
        g = a + np.mod(g - a, b - a)
        g.sort()
    return g


def _greville_tolerance(space: BsplineSpace1D) -> float:
    """Bound on the difference of two means of ``degree`` knots, each summed in its own order.

    Recursive summation of ``p`` terms has an error of at most ``(p - 1) u sum|x|`` (Higham,
    *Accuracy and Stability of Numerical Algorithms*, eq. 4.4, ``u = eps / 2``), so a mean is
    within ``(p - 1) u M + u |g|`` of the exact one with ``M = max|x|``, and two means differ by
    at most ``(p + 1) eps M``; one more ``eps M`` covers the wrap of a periodic space.
    """
    eps = _eps(space.knots.dtype.type)
    return (space.degree + 2) * eps * float(np.max(np.abs(space.knots)))


_OPEN_GREVILLE_SPACES: list[Callable[[type], BsplineSpace1D]] = [
    lambda t: BsplineSpace1D(np.array([0, 0, 0, 1, 2, 2, 3, 3, 3], dtype=t), 2),
    lambda t: BsplineSpace1D(np.array([0, 1, 3], dtype=t), 0),
    lambda t: BsplineSpace1D(np.array([0, 0, 0.5, 0.75, 1, 1], dtype=t), 1),
    lambda t: BsplineSpace1D(
        create_uniform_open_knots(6, 3, 1, domain=(t(1000.0), t(1007.0)), dtype=t), 3
    ),
    lambda t: BsplineSpace1D(
        create_uniform_open_knots(9, 5, 2, domain=(t(-3.0), t(5.0)), dtype=t), 5
    ),
    lambda t: BsplineSpace1D(
        create_uniform_open_knots(40, 9, 8, domain=(t(-1e-3), t(2e-3)), dtype=t), 9
    ),
]


@pytest.mark.parametrize("make", _OPEN_GREVILLE_SPACES, ids=range(len(_OPEN_GREVILLE_SPACES)))
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
def test_greville_of_open_spaces_matches_numpy_means(
    dtype: type, make: Callable[[type], BsplineSpace1D]
) -> None:
    """The abscissae are ``np.mean`` of ``degree`` consecutive knots (midpoints at degree 0)."""
    space = make(dtype)

    got = get_greville_abscissae(space)
    want = _greville_by_definition(space)

    assert got.dtype == dtype
    assert got.shape == want.shape == (space.num_basis,)
    assert np.all(np.abs(got - want) <= _greville_tolerance(space))


@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize(("n", "p"), [(4, 2), (6, 3), (6, 5), (8, 1), (5, 4)])
def test_greville_of_a_periodic_cardinal_space_wraps_into_the_domain_and_sorts(
    n: int, p: int, dtype: type
) -> None:
    """Cardinal knots are integers, so every mean is exactly representable and no wrap is a tie.

    The abscissae of the periodic space are ``np.mod`` of the means into ``[0, n)``, sorted: all
    inside the domain, increasing, and equal to the NumPy definition exactly.
    """
    space = BsplineSpace1D(create_cardinal_knots(n, p, dtype), p, periodic=True)

    got = get_greville_abscissae(space)
    want = _greville_by_definition(space)

    np.testing.assert_array_equal(got, want)
    assert np.all(got >= 0)
    assert np.all(got < n)
    assert np.all(np.diff(got) > 0)


def test_greville_hand_values() -> None:
    """Worked out by hand: means of two knots for the open quadratic, the wrapped cardinal."""
    open_quadratic = BsplineSpace1D([0, 0, 0, 1, 2, 2, 3, 3, 3], 2)
    np.testing.assert_array_equal(
        get_greville_abscissae(open_quadratic), [0.0, 0.5, 1.5, 2.0, 2.5, 3.0]
    )
    # Cardinal quadratic on [0, 4]: the means -0.5, 0.5, 1.5, 2.5; -0.5 wraps to 3.5 and sorts last.
    cardinal = BsplineSpace1D(create_cardinal_knots(4, 2), 2, periodic=True)
    np.testing.assert_array_equal(get_greville_abscissae(cardinal), [0.5, 1.5, 2.5, 3.5])
    # A mean exactly on the end belongs to [a, b) at a: knots [0, 0, 1, 2, 2], degree 1.
    on_the_end = BsplineSpace1D([0, 0, 1, 2, 2], 1, periodic=True)
    np.testing.assert_array_equal(get_greville_abscissae(on_the_end), [0.0, 0.0, 1.0])


def test_greville_of_negative_zero_knots_equals_zero() -> None:
    """A mean of two ``-0.0`` knots is a zero (the sign varies with NumPy: not asserted)."""
    space = BsplineSpace1D([-1, -1, -1, -0.0, -0.0, 1, 1, 1], 2)

    np.testing.assert_array_equal(get_greville_abscissae(space), [-1.0, -0.5, 0.0, 0.5, 1.0])


# ---------------------------------------------------------------------------
# Pickle and copy
# ---------------------------------------------------------------------------

_NEAR_HALF = float(np.nextafter(0.5, 1.0)) + 4 * float(np.spacing(0.5))
"""A knot 5 ulp above 0.5: inside the tolerance of 0.5 (8 eps), so the pair is one knot."""


def _spaces_1d() -> dict[str, BsplineSpace1D]:
    """Spaces whose state is not a function of the constructor arguments alone."""
    return {
        "open-f64": BsplineSpace1D(create_uniform_open_knots(4, 3), 3),
        "open-f32": BsplineSpace1D(
            create_uniform_open_knots(3, 2, domain=(np.float32(-1.0), np.float32(2.0))), 2
        ),
        "periodic": BsplineSpace1D(create_cardinal_knots(5, 2), 2, periodic=True),
        "degree-0": BsplineSpace1D([0, 1, 3, 4], 0),
        "snapped": BsplineSpace1D([0, 0, 0, 0.5, _NEAR_HALF, 1, 1, 1], 2),
        "unsnapped": BsplineSpace1D([0, 0, 0, 0.5, _NEAR_HALF, 1, 1, 1], 2, snap_knots=False),
    }


def _assert_same_space(got: BsplineSpace1D, want: BsplineSpace1D) -> None:
    """Assert two 1D spaces are the same value: state, derived quantities and basis."""
    assert type(got) is type(want)
    assert got.dtype == want.dtype
    assert got.degree == want.degree
    assert got.periodic == want.periodic
    assert got.tolerance == want.tolerance  # the stored tolerance, exactly
    assert got.num_basis == want.num_basis
    assert got.num_intervals == want.num_intervals
    np.testing.assert_array_equal(got.knots, want.knots)  # includes unsnapped values
    assert not got.knots.flags.writeable
    a, b = (float(v) for v in want.domain)
    pts = np.linspace(a, b, 23).astype(want.dtype)
    got_values, got_first = got.tabulate_basis(pts)
    want_values, want_first = want.tabulate_basis(pts)
    np.testing.assert_array_equal(got_values, want_values)
    np.testing.assert_array_equal(got_first, want_first)


def _pickled(protocol: int) -> Callable[[Any], Any]:
    """Return a round trip through ``pickle`` at ``protocol``."""
    return lambda x: pickle.loads(pickle.dumps(x, protocol=protocol))


_COPIERS = [
    pytest.param(copy.copy, id="copy"),
    pytest.param(copy.deepcopy, id="deepcopy"),
    *[pytest.param(_pickled(p), id=f"pickle-{p}") for p in range(pickle.HIGHEST_PROTOCOL + 1)],
]


@pytest.mark.parametrize("copier", _COPIERS)
@pytest.mark.parametrize("name", list(_spaces_1d()))
def test_a_1d_space_survives_copy_and_pickle_by_value(
    name: str, copier: Callable[[Any], Any]
) -> None:
    """Knots (snapped or as supplied), tolerance and basis are the original's.

    Catches: a rebuild that re-derives the tolerance or re-snaps (the ``unsnapped`` space would
    come back snapped, the ``snapped`` one with a tolerance of its snapped knots).
    """
    space = _spaces_1d()[name]

    _assert_same_space(copier(space), space)


def test_the_unsnapped_space_really_differs_from_a_snapped_one() -> None:
    """Premise of the previous test: the two near-duplicate spaces store different knots."""
    spaces = _spaces_1d()
    assert spaces["unsnapped"].knots[4] == _NEAR_HALF
    assert spaces["snapped"].knots[4] == 0.5
    assert _NEAR_HALF != 0.5


@pytest.mark.parametrize("copier", _COPIERS)
def test_a_tensor_product_space_survives_copy_and_pickle_by_value(
    copier: Callable[[Any], Any],
) -> None:
    """The directions, their tolerances and the tabulated basis come back unchanged."""
    spaces = _spaces_1d()
    space = BsplineSpace([spaces["open-f64"], spaces["snapped"]])

    got = copier(space)

    assert isinstance(got, BsplineSpace)
    assert got.dim == space.dim
    assert got.dtype == space.dtype
    assert got.tolerance == space.tolerance
    for g, w in zip(got.spaces, space.spaces, strict=True):
        _assert_same_space(g, w)
    pts = np.array([[0.0, 0.0], [0.3, 0.5], [0.77, 0.25], [1.0, 1.0]])
    got_values, got_first = got.tabulate_basis(pts)
    want_values, want_first = space.tabulate_basis(pts)
    np.testing.assert_array_equal(got_values, want_values)
    np.testing.assert_array_equal(got_first, want_first)


class _Tagged(BsplineSpace1D):
    """A 1D subclass whose instances carry an attribute of their own, as a user's might."""

    tag: str


class _TaggedND(BsplineSpace):
    """The same for the tensor product."""

    tag: str


def _tagged_1d() -> BsplineSpace1D:
    """Return a ``_Tagged`` space carrying its attribute."""
    space = _Tagged([0, 0, 0, 1, 1, 1], 2)
    space.tag = "kept"
    return space


def _tagged_nd() -> BsplineSpace:
    """Return a ``_TaggedND`` space carrying its attribute."""
    line = BsplineSpace1D([0, 0, 0, 1, 1, 1], 2)
    space = _TaggedND([line, line])
    space.tag = "kept"
    return space


@pytest.mark.parametrize("make", [_tagged_1d, _tagged_nd], ids=["1d", "nd"])
@pytest.mark.parametrize("copier", _COPIERS)
def test_a_subclass_keeps_its_instance_attributes_through_copy_and_pickle(
    copier: Callable[[Any], Any], make: Callable[[], Any]
) -> None:
    """A copy is of the same class and carries the attributes the instance had."""
    space = make()

    got = copier(space)

    assert type(got) is type(space)
    assert got.tag == "kept"


def _pickle_as_previous_release(cls: type, attributes: dict[str, Any], protocol: int) -> bytes:
    """Pickle what the previous release's ``cls`` pickled to: a plain object with these attributes.

    A stand-in class of the same module and name is put in the module for the dump, so the default
    pickling writes ``module.name`` plus a state dict, as the old class did; the real class is
    back in place before anything is loaded.

    Args:
        cls (type): The real class.
        attributes (dict[str, Any]): The attributes the old instance carried.
        protocol (int): The pickle protocol.

    Returns:
        bytes: The pickle.
    """
    module = sys.modules[cls.__module__]
    stand_in = type(cls.__name__, (), {"__module__": cls.__module__})
    obj = stand_in()
    obj.__dict__.update(attributes)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(module, cls.__name__, stand_in)
        return pickle.dumps(obj, protocol=protocol)


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
def test_a_1d_space_pickled_by_the_previous_release_loads_with_its_stored_state(
    dtype: type, protocol: int
) -> None:
    """The four attributes the old class stored rebuild an equal space, tolerance included."""
    knots: _Array = np.array([0, 0, 0, 0.25, 0.5, 1, 1, 1], dtype=dtype)
    built = BsplineSpace1D(knots, 2)
    data = _pickle_as_previous_release(
        BsplineSpace1D,
        {"_knots": built.knots.copy(), "_degree": 2, "_periodic": False, "_tol": built.tolerance},
        protocol,
    )

    loaded = pickle.loads(data)

    _assert_same_space(loaded, built)


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_a_stored_tolerance_is_used_not_derived_again(protocol: int) -> None:
    """The old state's ``_tol`` wins over the one the knots would give, and the knots are kept.

    With ``_tol = 0.3`` the knots 1 and 1.25 are one knot (two intervals, not three), and the
    near-duplicate pair is not snapped on load.
    """
    knots = np.array([0, 0, 0, 1, 1.25, 2, 2, 2])
    assert BsplineSpace1D(knots, 2).num_intervals == 3
    data = _pickle_as_previous_release(
        BsplineSpace1D,
        {"_knots": knots.copy(), "_degree": 2, "_periodic": False, "_tol": 0.3},
        protocol,
    )

    loaded = pickle.loads(data)

    assert loaded.tolerance == 0.3
    assert loaded.num_intervals == 2
    np.testing.assert_array_equal(loaded.knots, knots)


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_a_periodic_1d_space_pickled_by_the_previous_release_loads(protocol: int) -> None:
    """The periodic flag is restored, not reset: the count of functions depends on it."""
    knots = create_cardinal_knots(5, 2)
    built = BsplineSpace1D(knots, 2, periodic=True)
    data = _pickle_as_previous_release(
        BsplineSpace1D,
        {"_knots": knots.copy(), "_degree": 2, "_periodic": True, "_tol": built.tolerance},
        protocol,
    )

    loaded = pickle.loads(data)

    assert loaded.periodic
    assert loaded.num_basis == built.num_basis == 5
    _assert_same_space(loaded, built)


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_a_tensor_product_space_pickled_by_the_previous_release_loads(protocol: int) -> None:
    """The old ``_spaces`` tuple (with cached properties alongside) rebuilds a working space."""
    spaces = _spaces_1d()
    built = BsplineSpace([spaces["open-f64"], spaces["periodic"]])
    data = _pickle_as_previous_release(
        BsplineSpace,
        {"_spaces": built.spaces, "degrees": built.degrees, "tolerance": built.tolerance},
        protocol,
    )

    loaded = pickle.loads(data)

    assert isinstance(loaded, BsplineSpace)
    assert loaded.dim == 2
    assert loaded.degrees == built.degrees
    assert loaded.num_basis == built.num_basis
    pts = np.array([[0.1, 0.2], [0.9, 3.5]])
    got_values, got_first = loaded.tabulate_basis(pts)
    want_values, want_first = built.tabulate_basis(pts)
    np.testing.assert_array_equal(got_values, want_values)
    np.testing.assert_array_equal(got_first, want_first)


class _NamedND(BsplineSpace):
    """A tensor-product subclass whose constructor takes another argument, as a user's might."""

    name: str

    def __init__(self, spaces: list[BsplineSpace1D], name: str) -> None:
        """Build the space and keep its name.

        Args:
            spaces (list[BsplineSpace1D]): The directions.
            name (str): A label.
        """
        super().__init__(spaces)
        self.name = name


def test_a_subclass_with_its_own_constructor_signature_round_trips_through_pickle() -> None:
    """Unpickling does not call the subclass's constructor (it may need other arguments)."""
    line = BsplineSpace1D([0, 0, 0, 0.5, 1, 1, 1], 2)
    space = _NamedND([line, line], "kept")

    got = pickle.loads(pickle.dumps(space))

    assert type(got) is _NamedND
    assert got.name == "kept"
    assert got.num_basis == space.num_basis


def test_every_cached_property_is_part_of_the_value_not_extra_state() -> None:
    """After every cached property is computed, a plain space pickles no extra attribute.

    A cached property missing from the value attributes would be pickled as subclass state and
    restored over the rebuilt value, possibly stale.
    """
    line = BsplineSpace1D([0, 0, 0, 0.5, 1, 1, 1], 2)
    _ = line.domain, line.get_unique_knots_and_multiplicity(), line.first_basis_per_interval()
    _ = line.get_unique_knots_and_multiplicity(in_domain=True)
    space = BsplineSpace([line, line])
    _ = (space.degrees, space.tolerance, space.num_basis, space.num_total_basis)
    _ = (space.num_intervals, space.num_total_intervals, space.domain)
    cached_1d = {k for k, v in vars(type(line)).items() if isinstance(v, functools.cached_property)}
    cached_nd = {
        k for k, v in vars(type(space)).items() if isinstance(v, functools.cached_property)
    }

    assert cached_1d <= set(vars(line))
    assert cached_nd <= set(vars(space))
    assert line.__reduce__()[2] == {}
    assert space.__reduce__()[2] == {}


@pytest.mark.parametrize("dtype", _DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize(
    "knots",
    [
        [0, 0, 0, 0.5, 0.5 + 1e-17, 1, 1, 1],
        [-0.0, 0.0, 0, 0.25, 1, 1, 1],
        [1e3, 1e3, 1e3, 1e3 + 0.5, 1e3 + 1, 1e3 + 1, 1e3 + 1],
        [0, 1, 2, 3, 4, 5, 6, 7, 10],
    ],
)
def test_a_space_rebuilt_from_its_stored_state_equals_it(dtype: type, knots: list[float]) -> None:
    """The stored knots and tolerance rebuild the same classes, intervals and first indices.

    This is the property the single classification of a snapped vector rests on: re-classifying
    the snapped knots at the tolerance the space was built with gives the space's own classes.
    """
    space = BsplineSpace1D(np.array(knots, dtype), 2)

    rebuilt = pickle.loads(pickle.dumps(space))

    np.testing.assert_array_equal(rebuilt.knots, space.knots)
    assert rebuilt.tolerance == space.tolerance
    for in_domain in (False, True):
        got = rebuilt.get_unique_knots_and_multiplicity(in_domain=in_domain)
        want = space.get_unique_knots_and_multiplicity(in_domain=in_domain)
        np.testing.assert_array_equal(got[0], want[0])
        np.testing.assert_array_equal(got[1], want[1])
    assert rebuilt.num_intervals == space.num_intervals
    np.testing.assert_array_equal(
        rebuilt.first_basis_per_interval(), space.first_basis_per_interval()
    )


@pytest.mark.parametrize("tol", [-1.0, float("nan")])
def test_a_stored_state_with_an_impossible_tolerance_is_refused(tol: float) -> None:
    """A corrupt pickle's negative or NaN tolerance is refused rather than splitting every knot."""
    with pytest.raises(ValueError, match="stored tolerance must be non-negative"):
        BsplineSpace1D._from_state(np.array([0.0, 0.0, 1.0, 1.0]), 1, False, tol)
