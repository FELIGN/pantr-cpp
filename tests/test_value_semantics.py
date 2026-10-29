"""Value semantics of the types ported to immutable C++ values.

``AABB``, ``AffineTransform`` and ``Bezier`` hold a compiled value that Python can neither copy nor
pickle by itself. What their wrappers promise on top of it (docs/changelog.md, Unreleased) is
pinned here, one regression per test:

* ``Bezier`` copies its control points in, hands them out read-only, and replaces the value it holds
  on ``in_place=True`` instead of writing into it;
* ``Bezier`` stores ``float32`` as ``float32`` and every other real dtype as ``float64``;
* all three pickle and copy by value, a subclass keeps its class and attributes, and a ``Bezier``
  or ``AffineTransform`` pickled by the previous release still loads;
* ``copy=False`` conversions from a B-spline share its array only when that is safe, and keep it
  alive;
* ``AffineTransform`` leaves the caller's arrays alone, refuses absurd dimensions, and applies to
  points of any leading shape at any thread count;
* ``AABB.union`` returns the non-empty operand itself.

Oracles are NumPy and closed forms (``flip``, ``transpose``, ``@``, the Bernstein form), never the
wrappers' own helpers. Equality is bitwise only for data that is copied, not computed.
"""

from __future__ import annotations

import copy
import gc
import math
import pickle
import subprocess
import sys
import textwrap
import threading
import weakref
from collections.abc import Callable
from types import ModuleType
from typing import Any

import numba
import numpy as np
import numpy.typing as npt
import pytest

import pantr
import pantr.bezier._bezier as bezier_module
import pantr.transform as transform_module
from pantr.bezier import Bezier, create_from_bspline
from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D
from pantr.geometry import AABB
from pantr.transform import AffineTransform
from tests._tree_env import subprocess_env

_Array = npt.NDArray[np.float32 | np.float64]

# ---------------------------------------------------------------------------
# Helpers and fixtures
# ---------------------------------------------------------------------------


def _assert_within(got: npt.ArrayLike, want: npt.ArrayLike, tol: npt.ArrayLike) -> None:
    """Assert ``|got - want| <= tol`` entrywise, reporting the worst entry (``tol`` is an array).

    Args:
        got (npt.ArrayLike): The result under test.
        want (npt.ArrayLike): The oracle's value.
        tol (npt.ArrayLike): The absolute tolerance, broadcast against the difference.
    """
    err = np.abs(np.asarray(got, dtype=np.float64) - np.asarray(want, dtype=np.float64))
    bound = np.broadcast_to(np.asarray(tol, dtype=np.float64), err.shape)
    assert err.shape == np.shape(want), f"shape {err.shape} against {np.shape(want)}"
    excess = err - bound
    worst = int(np.argmax(excess))
    assert (err <= bound).all(), (
        f"{int((err > bound).sum())} of {err.size} entries beyond tolerance; worst entry "
        f"{worst} has error {err.flat[worst]:.3e} against {bound.flat[worst]:.3e}"
    )


def _curve(dtype: npt.DTypeLike = np.float64) -> _Array:
    """Build the control points of a fresh cubic curve in the plane (caller-owned, writeable)."""
    return np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 1.0], [4.0, 4.0]], dtype=dtype)


def _surface(dtype: npt.DTypeLike = np.float64) -> _Array:
    """Build the control points of a fresh (2, 3)-degree surface of rank 2, all entries distinct."""
    return (np.arange(24).reshape(3, 4, 2) * 0.5 + 1.0).astype(dtype)


def _volume(dtype: npt.DTypeLike = np.float64) -> _Array:
    """Build the control points of a fresh (1, 2, 3)-degree volume of rank 2, entries distinct."""
    return (np.arange(48).reshape(2, 3, 4, 2) * 0.5 + 1.0).astype(dtype)


def _bernstein(coeffs: npt.NDArray[Any], t: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    """Evaluate ``sum_i C(p, i) t^i (1 - t)^(p - i) coeffs[i]``: the closed form of a curve.

    Args:
        coeffs (npt.NDArray[Any]): The ``(p + 1, rank)`` coefficients.
        t (npt.NDArray[Any]): The parameters.

    Returns:
        npt.NDArray[np.float64]: The ``(len(t), rank)`` values.
    """
    p = coeffs.shape[0] - 1
    t64 = np.asarray(t, dtype=np.float64)
    basis = np.stack([math.comb(p, i) * t64**i * (1.0 - t64) ** (p - i) for i in range(p + 1)], 1)
    out: npt.NDArray[np.float64] = basis @ coeffs
    return out


def _assert_curve_matches(b: Bezier, cp: _Array, dtype: npt.DTypeLike) -> None:
    """Assert ``b.evaluate`` and its first derivative equal the closed form for ``cp``.

    The Bernstein basis is non-negative and sums to one; each value is a sum of p + 1 products of at
    most p + 1 rounded factors, so the error is bounded by about (p + 2) eps sum_i B_i |c_i|. The
    derivative is the same sum over p (c_{i+1} - c_i), whose differences add one more rounding
    relative to |c_i| + |c_{i+1}|. A factor 4 covers the scheme, the build and the float64 oracle.

    Args:
        b (Bezier): The curve under test (rank 1 or more, any degree).
        cp (_Array): Its control points, ``(p + 1, rank)``, as the oracle's input.
        dtype (npt.DTypeLike): The dtype ``b`` must work in (and the points are given in).
    """
    p = cp.shape[0] - 1
    eps = float(np.finfo(dtype).eps)
    t = np.linspace(0.0, 1.0, 9).astype(dtype)
    c64 = cp.astype(np.float64)
    want = _bernstein(c64, t)
    tol = 4 * (p + 2) * eps * _bernstein(np.abs(c64), t)
    _assert_within(b.evaluate(t).reshape(want.shape), want, tol)
    d_want = _bernstein(p * np.diff(c64, axis=0), t)
    d_mag = _bernstein(p * (np.abs(c64[1:]) + np.abs(c64[:-1])), t)
    d_got = b.evaluate_derivatives(t, 1).reshape(d_want.shape)
    _assert_within(d_got, d_want, 8 * (p + 2) * eps * d_mag)


def _assert_matches_matrix_product(
    got: npt.NDArray[np.float64],
    matrix: npt.NDArray[np.float64],
    offset: npt.NDArray[np.float64],
    pts: npt.NDArray[np.float64],
) -> None:
    """Assert ``got`` equals ``pts @ matrix.T + offset`` in shape, dtype and values.

    Each output is a sum of n products and the offset, n + 1 terms. Summed in any order a
    computation is within n u S of exact, S the sum of the terms' magnitudes and u = eps / 2, so
    two computations (the C++ loop, NumPy's matmul) differ by at most n eps S. The bound used is
    2 (n + 1) eps S: the factor 2 covers the rounding of S and FMA or vector-width variation.

    Args:
        got (npt.NDArray[np.float64]): The result under test.
        matrix (npt.NDArray[np.float64]): The ``(n, n)`` matrix of the map.
        offset (npt.NDArray[np.float64]): The ``(n,)`` offset of the map.
        pts (npt.NDArray[np.float64]): The points, ``(..., n)``.
    """
    n = matrix.shape[0]
    assert got.shape == pts.shape
    assert got.dtype == np.float64
    magnitude = np.abs(pts) @ np.abs(matrix).T + np.abs(offset)
    tol = 2 * (n + 1) * float(np.finfo(np.float64).eps) * magnitude
    _assert_within(got, pts @ matrix.T + offset, tol)


# ---------------------------------------------------------------------------
# Bezier: construction copies, control_points is read-only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_input",
    [
        pytest.param(lambda: _curve(np.float64), id="curve-f64"),
        pytest.param(lambda: _curve(np.float32), id="curve-f32"),
        pytest.param(lambda: _surface(np.float64), id="surface-f64"),
        pytest.param(lambda: _volume(np.float64), id="volume-f64"),
        pytest.param(lambda: np.array([[1.0, 2.0]]), id="degree-0"),
    ],
)
def test_constructor_copies_the_control_points(make_input: Callable[[], _Array]) -> None:
    """A later write to the input must not reach the Bézier (the changelog's first promise)."""
    cp = make_input()
    expected = cp.copy()

    b = Bezier(cp)
    cp[...] = -123.0  # before the first read of `control_points`, which is cached lazily

    np.testing.assert_array_equal(b.control_points, expected)
    assert not np.shares_memory(b.control_points, cp)


def test_constructor_copies_a_one_dimensional_input() -> None:
    """The ``(n,)`` input is viewed as ``(n, 1)`` in Python before it reaches the copy."""
    cp = np.array([0.0, 1.0, 4.0])
    b = Bezier(cp)
    cp[...] = -123.0
    np.testing.assert_array_equal(b.control_points, [[0.0], [1.0], [4.0]])


@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["f64", "f32"])
def test_evaluation_ignores_later_writes_to_the_constructor_input(dtype: type) -> None:
    """The compiled value must hold its own data, not only the cached ``control_points`` view."""
    cp = _curve(dtype)
    expected = cp.copy()
    b = Bezier(cp)
    cp[...] = -123.0
    _assert_curve_matches(b, expected, dtype)


@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["f64", "f32"])
def test_control_points_are_read_only(dtype: type) -> None:
    """A write through ``control_points`` raises, and leaves the values alone."""
    original = _curve(dtype)
    cp = Bezier(original.copy()).control_points
    assert not cp.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        cp[0, 0] = 7.0
    with pytest.raises(ValueError, match="read-only"):
        cp[...] = 0.0
    np.testing.assert_array_equal(cp, original)


@pytest.mark.parametrize("read_first", [False, True], ids=["unread", "already-read"])
def test_control_points_refuse_assignment(read_first: bool) -> None:
    """``b.control_points = x`` raises AttributeError, whether or not the view was cached yet."""
    original = _curve()
    b = Bezier(original.copy())
    cached = b.control_points if read_first else None

    with pytest.raises(AttributeError, match="control_points"):
        b.control_points = np.zeros_like(original)

    np.testing.assert_array_equal(b.control_points, original)
    if read_first:
        assert b.control_points is cached


@pytest.mark.parametrize("read_first", [False, True], ids=["unread", "already-read"])
def test_control_points_refuse_deletion(read_first: bool) -> None:
    """``del b.control_points`` raises AttributeError (a cached property lets it through)."""
    original = _curve()
    b = Bezier(original.copy())
    cached = b.control_points if read_first else None

    with pytest.raises(AttributeError, match="control_points"):
        del b.control_points

    np.testing.assert_array_equal(b.control_points, original)
    if read_first:
        assert b.control_points is cached


# ---------------------------------------------------------------------------
# Bezier: reverse, and the in_place operations replace the value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("direction", [0, 1])
def test_reverse_returns_an_independent_bezier(direction: int) -> None:
    """``reverse()`` shared memory with the original (``np.flip``); it must be a separate value."""
    cp = _surface()
    b = Bezier(cp.copy())

    r = b.reverse(direction)

    assert r is not b
    np.testing.assert_array_equal(r.control_points, np.flip(cp, axis=direction))
    assert not np.shares_memory(r.control_points, b.control_points)
    # An in-place change of either must not show in the other. The control points are half-integers
    # below 2^6, a translation adds 10 and 20, and the identity part multiplies by 1 and 0: all
    # exact in binary, so the comparison may be bitwise.
    r.transform(AffineTransform.translation([10.0, 20.0]), in_place=True)
    np.testing.assert_array_equal(b.control_points, cp)
    b.permute_directions([1, 0], in_place=True)
    shifted = np.flip(cp, axis=direction) + np.array([10.0, 20.0])
    np.testing.assert_array_equal(r.control_points, shifted)


# name -> (operation, expected control points of the volume, expected degrees). The volume has
# degrees (1, 2, 3) and rank 2; the control points only move, so the expected arrays are exact.
# The two cyclic permutations are each other's inverse: an involution such as [1, 0] could not
# tell a permutation from its inverse. New direction k is old direction permutation[k].
_MOVES: dict[str, tuple[Callable[[Bezier], None], Callable[[_Array], _Array], tuple[int, ...]]] = {
    "reverse": (
        lambda b: b.reverse(1, in_place=True),
        lambda cp: np.flip(cp, axis=1),
        (1, 2, 3),
    ),
    "permute-120": (
        lambda b: b.permute_directions([1, 2, 0], in_place=True),
        lambda cp: np.transpose(cp, (1, 2, 0, 3)),
        (2, 3, 1),
    ),
    "permute-201": (
        lambda b: b.permute_directions([2, 0, 1], in_place=True),
        lambda cp: np.transpose(cp, (2, 0, 1, 3)),
        (3, 1, 2),
    ),
}

# Entries exact in float32 and float64: the matrix is cast to the control points' type or not,
# either way the same map.
_MATRIX = np.array([[0.0, -1.0], [2.0, 0.5]])
_OFFSET = np.array([3.0, -4.0])


@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["f64", "f32"])
@pytest.mark.parametrize("move", list(_MOVES))
def test_in_place_move_replaces_the_value_and_keeps_an_earlier_view(move: str, dtype: type) -> None:
    """An array read before ``in_place=True`` keeps its values; the object shows the new ones.

    A permute also changes the shape, so the cached degree and dim must follow; the cached dtype
    must survive.
    """
    operate, expected_of, expected_degree = _MOVES[move]
    cp = _volume(dtype)
    b = Bezier(cp.copy())
    earlier = b.control_points
    earlier_values = earlier.copy()

    assert operate(b) is None
    gc.collect()  # a view of a freed value would read garbage here

    expected = expected_of(cp)
    np.testing.assert_array_equal(earlier, earlier_values)
    np.testing.assert_array_equal(b.control_points, expected)
    assert b.control_points is not earlier
    assert b.degree == expected_degree
    assert b.dim == 3
    assert b.rank == 2
    assert b.control_points.dtype == dtype
    assert b.dtype == dtype


@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["f64", "f32"])
def test_in_place_transform_replaces_the_value_and_keeps_an_earlier_view(dtype: type) -> None:
    """Same promise for ``transform(in_place=True)``, whose result is computed, not moved.

    Tolerance: a coordinate is a sum of 2 products and the offset, 3 terms with magnitude sum S;
    the C++ loop and the float64 oracle each lie within 2 u S (u = eps / 2) of exact when the
    control points' type is the working type, so 2 * 3 * eps S bounds the difference with slack.
    """
    cp = _surface(dtype)
    b = Bezier(cp.copy())
    earlier = b.control_points
    earlier_values = earlier.copy()

    assert b.transform(AffineTransform(_MATRIX, _OFFSET), in_place=True) is None
    gc.collect()

    cp64 = cp.astype(np.float64)
    expected = cp64 @ _MATRIX.T + _OFFSET
    tol = 2 * 3 * float(np.finfo(dtype).eps) * (np.abs(cp64) @ np.abs(_MATRIX).T + np.abs(_OFFSET))
    np.testing.assert_array_equal(earlier, earlier_values)
    _assert_within(b.control_points, expected, tol)
    assert b.control_points is not earlier
    assert b.degree == (2, 3)
    assert b.control_points.dtype == dtype
    assert b.dtype == dtype


# ---------------------------------------------------------------------------
# Bezier: the dtype policy
# ---------------------------------------------------------------------------

# 0/1 entries are exact in every dtype below, booleans included.
_QUADRATIC = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])


def _strided(a: _Array) -> _Array:
    """Return ``a`` as a non-contiguous view of a wider array."""
    wide = np.zeros((a.shape[0], 2 * a.shape[1]), dtype=a.dtype)
    wide[:, ::2] = a
    view: _Array = wide[:, ::2]
    assert not view.flags.c_contiguous
    return view


_STORED = [
    pytest.param(lambda: _QUADRATIC.astype(np.float32), np.float32, id="float32"),
    pytest.param(lambda: np.asfortranarray(_QUADRATIC.astype(np.float32)), np.float32, id="f32-F"),
    pytest.param(lambda: _strided(_QUADRATIC.astype(np.float32)), np.float32, id="f32-strided"),
    pytest.param(lambda: _QUADRATIC.astype(">f4"), np.float32, id="f32-big-endian"),
    pytest.param(lambda: _QUADRATIC.astype(np.float64), np.float64, id="float64"),
    pytest.param(lambda: _QUADRATIC.astype(">f8"), np.float64, id="f64-big-endian"),
    pytest.param(lambda: _QUADRATIC.astype(np.int32), np.float64, id="int32"),
    pytest.param(lambda: _QUADRATIC.astype(np.int64), np.float64, id="int64"),
    pytest.param(lambda: _QUADRATIC.astype(np.uint8), np.float64, id="uint8"),
    pytest.param(lambda: _QUADRATIC.astype(">i4"), np.float64, id="int32-big-endian"),
    pytest.param(lambda: _QUADRATIC.astype(np.bool_), np.float64, id="bool"),
    pytest.param(lambda: _QUADRATIC.astype(np.float16), np.float64, id="float16"),
    pytest.param(lambda: _QUADRATIC.astype(np.longdouble), np.float64, id="longdouble"),
]


@pytest.mark.parametrize(("make_input", "stored"), _STORED)
def test_dtype_policy_storage(make_input: Callable[[], Any], stored: type) -> None:
    """float32 stays float32; every other real dtype is stored as native float64.

    Before the port a bool or float16 array was kept as given and failed in evaluation.
    """
    b = Bezier(make_input())

    assert b.dtype == stored
    assert b.control_points.dtype == stored
    assert b.control_points.dtype.isnative
    np.testing.assert_array_equal(b.control_points, _QUADRATIC)


@pytest.mark.parametrize(("make_input", "stored"), _STORED)
def test_dtype_policy_evaluation_results(make_input: Callable[[], Any], stored: type) -> None:
    """Evaluation returns arrays of the stored dtype, with the right values."""
    b = Bezier(make_input())
    t: npt.NDArray[Any] = np.linspace(0.0, 1.0, 9).astype(stored)

    assert b.evaluate(t).dtype == stored
    assert b.evaluate_derivatives(t, 1).dtype == stored
    _assert_curve_matches(b, _QUADRATIC, stored)


def test_dtype_policy_one_dimensional_float32_input() -> None:
    """The ``(n,)`` float32 input is viewed as ``(n, 1)`` and stays float32."""
    b = Bezier(np.array([0.0, 1.0, 4.0], dtype=np.float32))
    assert b.dtype == np.float32
    assert b.control_points.dtype == np.float32
    assert b.control_points.shape == (3, 1)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128], ids=["c64", "c128"])
@pytest.mark.parametrize("shape", [(3, 2), (3,)], ids=["2d", "1d"])
def test_complex_control_points_raise_type_error(dtype: type, shape: tuple[int, ...]) -> None:
    """Complex input is a TypeError, not a silent real part (or a crash in the binding)."""
    with pytest.raises(TypeError, match="real"):
        Bezier(np.ones(shape, dtype=dtype))


# ---------------------------------------------------------------------------
# Copy and pickle, all three classes
# ---------------------------------------------------------------------------


def _pickled(protocol: int) -> Callable[[Any], Any]:
    """Return a round trip through ``pickle`` at ``protocol``."""
    return lambda x: pickle.loads(pickle.dumps(x, protocol=protocol))


_PICKLES = [pytest.param(_pickled(p), id=f"pickle-{p}") for p in range(pickle.HIGHEST_PROTOCOL + 1)]
_DEEP_COPIERS = [pytest.param(copy.deepcopy, id="deepcopy"), *_PICKLES]
_COPIERS = [pytest.param(copy.copy, id="copy"), *_DEEP_COPIERS]


class _TaggedBezier(Bezier):
    """A subclass with attributes of its own, which copies and pickles must carry."""

    label: str
    history: list[Any]


class _TaggedTransform(AffineTransform):
    """A subclass with attributes of its own, which copies and pickles must carry."""

    label: str
    history: list[Any]


def _assert_read_only(arr: npt.NDArray[Any]) -> None:
    """Assert that ``arr`` is flagged read-only and refuses a write."""
    assert not arr.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        arr[...] = 0


@pytest.mark.parametrize("copier", _COPIERS)
@pytest.mark.parametrize(
    ("make_cp", "rational"),
    [
        pytest.param(_curve, False, id="curve-f64"),
        pytest.param(lambda: _curve(np.float32), True, id="rational-curve-f32"),
        pytest.param(_surface, False, id="surface-f64"),
        pytest.param(_volume, False, id="volume-f64"),
        pytest.param(lambda: np.array([[1.0, 2.0]]), False, id="degree-0"),
    ],
)
def test_bezier_round_trip_is_equal_by_value_and_read_only(
    copier: Callable[[Bezier], Bezier], make_cp: Callable[[], _Array], rational: bool
) -> None:
    """A copy equals the original by value, in dtype too, and its arrays are read-only."""
    cp = make_cp()
    original = Bezier(cp.copy(), is_rational=rational)

    got = copier(original)

    assert type(got) is Bezier
    assert got.is_rational is rational
    assert got.dtype == cp.dtype
    assert got.degree == tuple(s - 1 for s in cp.shape[:-1])
    assert got.rank == cp.shape[-1] - int(rational)
    np.testing.assert_array_equal(got.control_points, cp)
    assert got.control_points.dtype == cp.dtype
    _assert_read_only(got.control_points)


@pytest.mark.parametrize("copier", _COPIERS)
def test_bezier_round_trip_evaluates(copier: Callable[[Bezier], Bezier]) -> None:
    """The copied compiled value is usable, not only its cached attributes."""
    cp = _curve()
    got = copier(Bezier(cp.copy()))
    _assert_curve_matches(got, cp, np.float64)


@pytest.mark.parametrize("copier", _COPIERS)
def test_in_place_change_of_a_bezier_copy_leaves_the_original(
    copier: Callable[[Bezier], Bezier],
) -> None:
    """``copy.copy`` shares the compiled value, so ``in_place`` must replace, not write into it."""
    cp = _curve()
    original = Bezier(cp.copy())
    twin = copier(original)

    twin.reverse(in_place=True)

    np.testing.assert_array_equal(original.control_points, cp)
    np.testing.assert_array_equal(twin.control_points, np.flip(cp, axis=0))


@pytest.mark.parametrize("copier", _COPIERS)
@pytest.mark.parametrize("rational", [False, True], ids=["polynomial", "rational"])
def test_bezier_subclass_keeps_its_class_and_attributes(
    copier: Callable[[Bezier], Bezier], rational: bool
) -> None:
    """A subclass with attributes of its own survives copy, deepcopy and pickle as itself."""
    cp = _curve(np.float32)
    original = _TaggedBezier(cp.copy(), is_rational=rational)
    original.label = "left-wall"
    original.history = [1, [2, 3]]
    _ = original.control_points  # the cached view is derived, not state

    got = copier(original)

    assert type(got) is _TaggedBezier
    assert got.label == "left-wall"
    assert got.history == [1, [2, 3]]
    assert got.is_rational is rational
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got.control_points, cp)


@pytest.mark.parametrize("copier", _DEEP_COPIERS)
def test_bezier_deep_copy_does_not_share_a_mutable_attribute(
    copier: Callable[[Bezier], Bezier],
) -> None:
    """Deep means the subclass's own attributes are copied too."""
    original = _TaggedBezier(_curve())
    original.history = [1, [2, 3]]

    got = copier(original)

    assert isinstance(got, _TaggedBezier)
    assert got.history == original.history
    assert got.history is not original.history
    got.history[1].append(4)
    assert original.history == [1, [2, 3]]


def _pickle_as_previous_release(
    module: ModuleType, name: str, build: Callable[[type], object], protocol: int
) -> bytes:
    """Pickle what the previous release's class pickled to: a plain object with plain attributes.

    The stand-in is an ordinary class of the same module and qualified name, so the default
    pickling machinery writes the bytes the old class wrote (``module.name`` plus a state dict).
    The real class is back in place before anything is loaded.

    Args:
        module (ModuleType): The module defining the class.
        name (str): The class name.
        build (Callable[[type], object]): Builds the instance from the stand-in class.
        protocol (int): The pickle protocol.

    Returns:
        bytes: The pickle.
    """
    stand_in = type(name, (), {"__module__": module.__name__})
    obj = build(stand_in)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(module, name, stand_in)
        return pickle.dumps(obj, protocol=protocol)


def _plain(cls: type, **attrs: object) -> object:
    """Build an instance of ``cls`` with exactly the attributes ``attrs``, bypassing any init."""
    obj = cls()
    obj.__dict__.update(attrs)
    return obj


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["f64", "f32"])
def test_bezier_pickled_by_the_previous_release_loads_and_evaluates(
    protocol: int, dtype: type
) -> None:
    """The old plain attributes (``_control_points``, ``_is_rational``) rebuild the value."""
    cp = _curve(dtype)
    data = _pickle_as_previous_release(
        bezier_module,
        "Bezier",
        lambda cls: _plain(cls, _control_points=cp.copy(), _is_rational=False),
        protocol,
    )

    loaded = pickle.loads(data)

    assert type(loaded) is Bezier
    assert loaded.is_rational is False
    assert loaded.dtype == dtype
    assert loaded.degree == (3,)
    np.testing.assert_array_equal(loaded.control_points, cp)
    _assert_read_only(loaded.control_points)
    _assert_curve_matches(loaded, cp, dtype)
    loaded.reverse(in_place=True)  # the loaded object is a full value, not just readable
    np.testing.assert_array_equal(loaded.control_points, np.flip(cp, axis=0))


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_rational_bezier_pickled_by_the_previous_release_loads(protocol: int) -> None:
    """The old ``_is_rational=True`` state restores the flag, the rank and a working value.

    With unit weights the rational curve is the polynomial one in the first coordinate, whatever
    the convention for premultiplied weights; the Bernstein error bound of
    ``_assert_curve_matches`` (4 (p + 2) eps max|c|, with max|c| = 4 here) applies to the quotient
    by a denominator that is one to within a few eps.
    """
    cp = _curve()
    cp[:, 1] = 1.0
    data = _pickle_as_previous_release(
        bezier_module,
        "Bezier",
        lambda cls: _plain(cls, _control_points=cp.copy(), _is_rational=True),
        protocol,
    )

    loaded = pickle.loads(data)

    assert type(loaded) is Bezier
    assert loaded.is_rational is True
    assert loaded.rank == 1
    assert loaded.degree == (3,)
    np.testing.assert_array_equal(loaded.control_points, cp)
    t = np.linspace(0.0, 1.0, 9)
    want = _bernstein(cp[:, :1], t)[:, 0]
    atol = 2 * 4 * (3 + 2) * np.finfo(np.float64).eps * 4.0
    np.testing.assert_allclose(loaded.evaluate(t), want, rtol=0, atol=atol)


_OLD_MATRIX = np.array([[2.0, 1.0], [0.0, 3.0]])
_OLD_OFFSET = np.array([1.0, 2.0])
# Closed forms: A^-1 = [[1/2, -1/6], [0, 1/3]] and -A^-1 b = [-1/6, -2/3].
_OLD_INVERSE_MATRIX = np.array([[0.5, -1.0 / 6.0], [0.0, 1.0 / 3.0]])
_OLD_INVERSE_OFFSET = np.array([-1.0 / 6.0, -2.0 / 3.0])


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
@pytest.mark.parametrize("with_cached_inverse", [False, True], ids=["fresh", "inverse-cached"])
def test_affine_transform_pickled_by_the_previous_release_loads(
    protocol: int, with_cached_inverse: bool
) -> None:
    """The old plain attributes (``_matrix``, ``_translation``, a cached ``inverse``) load."""

    def build(cls: type) -> object:
        attrs: dict[str, object] = {"_matrix": _OLD_MATRIX.copy(), "_translation": _OLD_OFFSET}
        if with_cached_inverse:
            attrs["inverse"] = _plain(
                cls, _matrix=_OLD_INVERSE_MATRIX.copy(), _translation=_OLD_INVERSE_OFFSET.copy()
            )
        return _plain(cls, **attrs)

    data = _pickle_as_previous_release(transform_module, "AffineTransform", build, protocol)

    loaded = pickle.loads(data)

    assert type(loaded) is AffineTransform
    np.testing.assert_array_equal(loaded.matrix, _OLD_MATRIX)
    np.testing.assert_array_equal(loaded.offset, _OLD_OFFSET)
    _assert_read_only(loaded.matrix)
    pts = np.array([[1.0, 0.0], [0.0, 1.0], [-2.0, 5.0]])
    _assert_matches_matrix_product(loaded(pts), _OLD_MATRIX, _OLD_OFFSET, pts)
    # Each entry of the inverse and of its offset results from at most 3 rounded operations on
    # values of magnitude at most 1, so within 3 u = 1.5 eps; 8 eps leaves a factor of 5.
    eps = float(np.finfo(np.float64).eps)
    np.testing.assert_allclose(loaded.inverse.matrix, _OLD_INVERSE_MATRIX, rtol=0, atol=8 * eps)
    np.testing.assert_allclose(loaded.inverse.offset, _OLD_INVERSE_OFFSET, rtol=0, atol=8 * eps)


@pytest.mark.parametrize("copier", _COPIERS)
def test_affine_transform_round_trip_is_equal_by_value_and_read_only(
    copier: Callable[[AffineTransform], AffineTransform],
) -> None:
    """A copy has the original's matrix and offset (bitwise: copied data) and read-only arrays."""
    matrix = np.array([[0.0, -1.0, 0.5], [2.0, 0.25, 0.0], [1.0, 1.0, -3.0]])
    offset = np.array([3.0, -4.0, 0.125])
    original = AffineTransform(matrix.copy(), offset.copy())
    _ = original.inverse  # a cached inverse must not break the copy

    got = copier(original)

    assert type(got) is AffineTransform
    assert got.dim == 3
    np.testing.assert_array_equal(got.matrix, matrix)
    np.testing.assert_array_equal(got.offset, offset)
    _assert_read_only(got.matrix)
    _assert_read_only(got.offset)
    pts = np.array([[1.0, 2.0, 3.0], [-4.0, 0.5, 9.0]])
    _assert_matches_matrix_product(got(pts), matrix, offset, pts)


@pytest.mark.parametrize("copier", _COPIERS)
def test_affine_transform_subclass_keeps_its_class_and_attributes(
    copier: Callable[[AffineTransform], AffineTransform],
) -> None:
    """A subclass with attributes of its own survives copy, deepcopy and pickle as itself."""
    original = _TaggedTransform(np.array([[0.0, -1.0], [1.0, 0.0]]), np.array([1.0, 2.0]))
    original.label = "quarter-turn"
    original.history = [1, [2, 3]]
    _ = original.inverse

    got = copier(original)

    assert type(got) is _TaggedTransform
    assert got.label == "quarter-turn"
    assert got.history == [1, [2, 3]]
    np.testing.assert_array_equal(got.matrix, [[0.0, -1.0], [1.0, 0.0]])
    np.testing.assert_array_equal(got.offset, [1.0, 2.0])


@pytest.mark.parametrize("copier", _DEEP_COPIERS)
def test_affine_transform_deep_copy_does_not_share_a_mutable_attribute(
    copier: Callable[[AffineTransform], AffineTransform],
) -> None:
    """Deep means the subclass's own attributes are copied too."""
    original = _TaggedTransform(np.eye(2))
    original.history = [1, [2, 3]]

    got = copier(original)

    assert isinstance(got, _TaggedTransform)
    assert got.history == original.history
    assert got.history is not original.history


# name -> bounds, each pair is (lo, hi). Signed zeros and infinities must survive bit for bit.
_BOXES = {
    "finite-2d": ([0.0, -1.0], [2.0, 3.5]),
    "finite-3d": ([-1.0, 0.5, 2.0], [1.0, 0.75, 9.0]),
    "signed-zero": ([-0.0, 0.0], [0.0, -0.0]),
    "unbounded": ([-np.inf, -np.inf], [np.inf, np.inf]),
    "half-bounded": ([0.0, -np.inf], [np.inf, 1.0]),
    "empty": ([np.inf, np.inf], [-np.inf, -np.inf]),
    "empty-on-one-axis": ([0.0, 5.0], [1.0, 4.0]),
    "point": ([2.0, 3.0], [2.0, 3.0]),
}


@pytest.mark.parametrize("copier", _COPIERS)
@pytest.mark.parametrize("name", list(_BOXES))
def test_aabb_round_trip_is_equal_by_value_and_read_only(
    copier: Callable[[AABB], AABB], name: str
) -> None:
    """``pickle``, ``copy.copy`` and ``copy.deepcopy`` each raised AttributeError on an AABB."""
    lo, hi = _BOXES[name]
    original = AABB(lo, hi)

    got = copier(original)

    assert type(got) is AABB
    assert got == original
    assert hash(got) == hash(original)
    assert got.is_empty() == original.is_empty()
    # The corners are copied data, so compare the bytes: this also pins the sign of a zero.
    assert got.lo.tobytes() == np.array(lo, dtype=np.float64).tobytes()
    assert got.hi.tobytes() == np.array(hi, dtype=np.float64).tobytes()
    _assert_read_only(got.lo)
    _assert_read_only(got.hi)


# ---------------------------------------------------------------------------
# copy=False conversions from a B-spline
# ---------------------------------------------------------------------------


def _bspline_over(cp: _Array) -> Bspline:
    """Build the Bézier-like B-spline whose control points are ``cp`` (not copied if it fits)."""
    p = cp.shape[0] - 1
    knots = np.concatenate([np.zeros(p + 1), np.ones(p + 1)]).astype(cp.dtype)
    return Bspline(BsplineSpace([BsplineSpace1D(knots, p)]), cp)


def _via_to_bezier(bs: Bspline) -> Bezier:
    """Convert with ``Bspline.to_bezier(copy=False)``."""
    return bs.to_bezier(copy=False)


def _via_create_from_bspline(bs: Bspline) -> Bezier:
    """Convert with ``create_from_bspline(bs, copy=False)``."""
    return create_from_bspline(bs, copy=False)


_CONVERTERS = [
    pytest.param(_via_to_bezier, id="to_bezier"),
    pytest.param(_via_create_from_bspline, id="create_from_bspline"),
]
_DTYPES = [pytest.param(np.float64, id="f64"), pytest.param(np.float32, id="f32")]


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("convert", _CONVERTERS)
def test_copy_false_shares_a_contiguous_array_read_only(
    convert: Callable[[Bspline], Bezier], dtype: type
) -> None:
    """``copy=False`` on a C-contiguous float32/float64 array shares it, read-only to the Bézier."""
    cp = _curve(dtype)
    bs = _bspline_over(cp)

    bez = convert(bs)

    assert np.shares_memory(bez.control_points, bs.control_points)
    assert not bez.control_points.flags.writeable
    assert bez.dtype == dtype
    np.testing.assert_array_equal(bez.control_points, _curve(dtype))


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("convert", _CONVERTERS)
def test_copy_false_bezier_outlives_the_bspline_and_its_array(
    convert: Callable[[Bspline], Bezier], dtype: type
) -> None:
    """The Bézier keeps the shared array alive; dropping it (the last reference) releases it."""
    cp = _curve(dtype)
    expected = cp.copy()
    array_alive = weakref.ref(cp)
    bs = _bspline_over(cp)
    bez = convert(bs)
    del bs, cp
    gc.collect()

    assert array_alive() is not None  # a Bézier that did not hold it would have let it go
    # Allocate over whatever was freed, so a dangling pointer would read these instead.
    junk: list[npt.NDArray[Any]] = [
        np.full(expected.shape, -777.0, dtype=dtype) for _ in range(256)
    ]
    np.testing.assert_array_equal(bez.control_points, expected)
    _assert_curve_matches(bez, expected, dtype)

    del bez, junk
    gc.collect()
    assert array_alive() is None  # and a Bézier that held it for ever would leak it


@pytest.mark.parametrize("convert", _CONVERTERS)
def test_copy_false_bezier_dropped_on_another_thread(convert: Callable[[Bspline], Bezier]) -> None:
    """The last reference may go on a thread that is not the one that made the Bézier."""
    for _ in range(8):
        cp = _curve()
        array_alive = weakref.ref(cp)
        holders = [[convert(_bspline_over(cp))] for _ in range(4)]
        del cp
        gc.collect()
        threads = [threading.Thread(target=h.clear) for h in holders]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)
        assert not any(thread.is_alive() for thread in threads)
        gc.collect()
        assert array_alive() is None


def _reversed_view(a: _Array) -> _Array:
    return a[::-1]


def _fortran_order(a: _Array) -> _Array:
    return np.asfortranarray(a)


def _column_strided(a: _Array) -> _Array:
    wide: _Array = np.repeat(a, 2, axis=1)
    return wide[:, ::2]


@pytest.mark.parametrize(
    "layout",
    [
        pytest.param(_reversed_view, id="negative-stride"),
        pytest.param(_fortran_order, id="fortran-order"),
        pytest.param(_column_strided, id="strided-columns"),
    ],
)
@pytest.mark.parametrize("convert", _CONVERTERS)
def test_copy_false_does_not_share_an_array_of_another_layout(
    convert: Callable[[Bspline], Bezier], layout: Callable[[_Array], _Array]
) -> None:
    """A reversed view (say) is converted, so the Bézier shares nothing with the B-spline."""
    raw = _curve()
    view = layout(raw.copy())
    bs = _bspline_over(view)
    assert not bs.control_points.flags.c_contiguous  # the case is not vacuous
    seen = view.copy()

    bez = convert(bs)

    assert not np.shares_memory(bez.control_points, bs.control_points)
    np.testing.assert_array_equal(bez.control_points, seen)
    bs.control_points[...] = -777.0  # a write to the B-spline cannot reach the Bézier
    np.testing.assert_array_equal(bez.control_points, seen)


def test_to_bspline_copy_false_holds_the_read_only_view() -> None:
    """``Bezier.to_bspline(copy=False)`` shares the points and refuses an in-place change."""
    cp = _curve()
    b = Bezier(cp.copy())

    shared = b.to_bspline(copy=False)
    independent = b.to_bspline()

    assert np.shares_memory(shared.control_points, b.control_points)
    assert not shared.control_points.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        shared.reverse(in_place=True)
    assert not np.shares_memory(independent.control_points, b.control_points)
    np.testing.assert_array_equal(b.control_points, cp)


# ---------------------------------------------------------------------------
# AffineTransform
# ---------------------------------------------------------------------------


def _writeable_caller_arrays() -> tuple[npt.NDArray[np.float64], ...]:
    """Build a caller's float64 C-contiguous matrix, vector and center, all writeable."""
    return (
        np.array([[2.0, 0.5, 0.0], [0.0, 1.0, 0.25], [1.0, 0.0, 3.0]]),
        np.array([1.0, -2.0, 0.5]),
        np.array([0.5, 0.25, -1.0]),
    )


# name -> a call that is handed the caller's matrix, vector and center, in that order.
_BUILDERS: dict[str, Callable[..., AffineTransform]] = {
    "constructor": lambda m, v, c: AffineTransform(m, v),
    "constructor-matrix-only": lambda m, v, c: AffineTransform(m),
    "translation": lambda m, v, c: AffineTransform.translation(v),
    "scaling": lambda m, v, c: AffineTransform.scaling(v, center=c),
    "scaling-isotropic": lambda m, v, c: AffineTransform.scaling(2.0, center=c),
    "rotation_3d": lambda m, v, c: AffineTransform.rotation_3d(0.3, v, center=c),
    "rotation_2d": lambda m, v, c: AffineTransform.rotation_2d(0.3, center=c[:2]),
    "mirror": lambda m, v, c: AffineTransform.mirror(v, center=c),
}


@pytest.mark.parametrize("name", list(_BUILDERS))
def test_building_a_transform_leaves_the_callers_arrays_writeable(name: str) -> None:
    """The constructor flagged the caller's own float64 array read-only when it needed no cast."""
    matrix, vector, center = _writeable_caller_arrays()
    kept = [a.copy() for a in (matrix, vector, center)]

    _BUILDERS[name](matrix, vector, center)

    for arr, before in zip((matrix, vector, center), kept, strict=True):
        assert arr.flags.writeable
        np.testing.assert_array_equal(arr, before)
        arr[...] = arr  # and a write really goes through


def test_transform_keeps_its_own_copy_of_the_callers_arrays() -> None:
    """A later write to the matrix or vector handed in does not move the transform."""
    matrix, vector, _ = _writeable_caller_arrays()
    kept = (matrix.copy(), vector.copy())

    t = AffineTransform(matrix, vector)
    matrix[...] = -9.0
    vector[...] = -9.0

    np.testing.assert_array_equal(t.matrix, kept[0])
    np.testing.assert_array_equal(t.offset, kept[1])
    assert not np.shares_memory(t.matrix, matrix)
    assert not np.shares_memory(t.offset, vector)


# Run in a child process, with the address space capped once pantr is loaded: before the fix
# these calls crashed the interpreter, so a regression must fail a test and not end the session.
# 2 GiB is far above anything a validated call allocates and far below what a ValueError-less
# call would ask for.
_HUGE_DIM_CHILD = textwrap.dedent(
    """
    import resource
    from pantr.transform import AffineTransform

    cap = 2 << 30
    try:
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    except ValueError:
        # macOS refuses to lower RLIMIT_AS (EINVAL); a regression there still fails loudly, as
        # std::bad_alloc ("other") rather than "ValueError".
        pass
    try:
        {call}
    except ValueError:
        print("ValueError")
    except BaseException as exc:
        print("other:", type(exc).__name__)
    else:
        print("returned")
    """
)


@pytest.mark.parametrize(
    "call",
    [
        pytest.param("AffineTransform.identity(2**32)", id="identity"),
        pytest.param("AffineTransform.shear(2**32, 0, 1, 1.0)", id="shear"),
    ],
)
def test_absurd_dimension_raises_value_error(call: str) -> None:
    """``identity(2**32)`` and ``shear(2**32, 0, 1, 1.0)`` crashed; they are a ValueError."""
    proc = subprocess.run(
        [sys.executable, "-c", _HUGE_DIM_CHILD.format(call=call)],
        env=subprocess_env(),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    detail = f"exit {proc.returncode}; stdout {proc.stdout!r}; stderr tail {proc.stderr[-400:]!r}"
    assert proc.returncode == 0, detail
    assert proc.stdout.strip() == "ValueError", detail


_MAX_THREADS = int(numba.config.NUMBA_NUM_THREADS)
_THREADS = [
    pytest.param(1, id="serial"),
    pytest.param(min(8, _MAX_THREADS), id="parallel"),
]
# Leading shapes of the points; "pool" is large enough for the parallel path to engage.
_LEADS = [
    pytest.param((), id="one-point"),
    pytest.param((7,), id="k-points"),
    pytest.param((3, 5), id="a-by-b"),
    pytest.param((40, 1500), id="pool"),
]


def _as_layout(pts: npt.NDArray[np.float64], layout: str) -> npt.NDArray[np.float64]:
    """Return the values of ``pts`` in memory order ``layout``: C, Fortran, or last-axis strided."""
    if layout == "fortran":
        return np.asfortranarray(pts)
    if layout == "strided":
        wide = np.zeros((*pts.shape[:-1], 2 * pts.shape[-1]))
        wide[..., ::2] = pts
        view: npt.NDArray[np.float64] = wide[..., ::2]
        return view
    return pts


@pytest.mark.parametrize("layout", ["c", "fortran", "strided"])
@pytest.mark.parametrize("threads", _THREADS)
@pytest.mark.parametrize("lead", _LEADS)
@pytest.mark.parametrize("n", [2, 3])
def test_call_keeps_the_leading_shape_and_equals_the_matrix_product(
    n: int, lead: tuple[int, ...], threads: int, layout: str
) -> None:
    """``T(pts)`` has the shape of ``pts`` and equals ``pts @ A.T + b`` at any thread count."""
    rng = np.random.default_rng(20261029 + 100 * n + len(lead))
    matrix = rng.normal(size=(n, n))
    offset = 10.0 * rng.normal(size=n)
    pts = rng.uniform(-10.0, 10.0, size=(*lead, n))
    given = _as_layout(pts, layout)
    given_before = given.copy()
    t = AffineTransform(matrix.copy(), offset.copy())

    with pantr.num_threads(threads):
        got = t(given)

    _assert_matches_matrix_product(got, matrix, offset, pts)
    np.testing.assert_array_equal(given, given_before)


@pytest.mark.parametrize("lead", [(0,), (3, 0)], ids=["no-points", "empty-axis"])
def test_call_on_zero_points_keeps_the_shape(lead: tuple[int, ...]) -> None:
    """An empty batch returns an empty array of the same shape (no error, no flattening)."""
    got = AffineTransform.rotation_2d(0.3)(np.zeros((*lead, 2)))
    assert got.shape == (*lead, 2)
    assert got.dtype == np.float64


# ---------------------------------------------------------------------------
# AABB.union
# ---------------------------------------------------------------------------

_FULL_BOXES = [
    pytest.param(AABB([0.0, 0.0], [1.0, 1.0]), id="unit-square"),
    pytest.param(AABB([2.0, 3.0], [2.0, 3.0]), id="single-point"),
    pytest.param(AABB.unbounded(2), id="unbounded"),
    pytest.param(AABB([-np.inf, 0.0], [np.inf, 1.0]), id="strip"),
]
_EMPTY_BOXES = [
    pytest.param(AABB.empty(2), id="empty"),
    pytest.param(AABB([1.0, 1.0], [0.0, 0.0]), id="empty-by-hand"),
    pytest.param(AABB([0.0, 5.0], [1.0, 4.0]), id="empty-on-the-second-axis-only"),
]


@pytest.mark.parametrize("empty", _EMPTY_BOXES)
@pytest.mark.parametrize("full", _FULL_BOXES)
def test_union_with_an_empty_box_returns_the_other_operand_itself(full: AABB, empty: AABB) -> None:
    """The result is the very object, in both operand orders, not an equal copy."""
    assert not full.is_empty()
    assert empty.is_empty()
    assert full.union(empty) is full
    assert empty.union(full) is full


@pytest.mark.parametrize(
    ("a", "b"),
    [
        pytest.param(([0.0, 0.0], [1.0, 1.0]), ([2.0, -1.0], [3.0, 0.5]), id="disjoint"),
        pytest.param(([2.0, 3.0], [2.0, 3.0]), ([0.0, 0.0], [1.0, 1.0]), id="point-and-box"),
        pytest.param(([0.0, 0.0], [2.0, 1.0]), ([1.0, 0.5], [3.0, 4.0]), id="overlapping"),
    ],
)
def test_union_of_two_non_empty_boxes_is_the_componentwise_hull(
    a: tuple[list[float], list[float]], b: tuple[list[float], list[float]]
) -> None:
    """A non-empty point box is not 'empty': the union is the hull, not either operand."""
    first, second = AABB(*a), AABB(*b)

    u = first.union(second)

    np.testing.assert_array_equal(u.lo, np.minimum(a[0], b[0]))
    np.testing.assert_array_equal(u.hi, np.maximum(a[1], b[1]))
    assert u is not first
    assert u is not second
