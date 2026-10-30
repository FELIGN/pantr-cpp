"""Value semantics of ``Bspline``, now an immutable value over the C++ library.

``Bspline`` holds a compiled value that Python can neither copy nor pickle by itself. What its
wrapper promises on top of it (docs/changelog.md, Unreleased) is pinned here, one regression per
test:

* the constructor copies the control points in, in the order count, dtype, rank, and
  ``control_points`` is read-only and refuses assignment;
* ``in_place=True`` replaces the value the object holds (the space too) instead of writing into
  it, and an array read before keeps the old values;
* ``copy``, ``deepcopy`` and ``pickle`` copy by value, a subclass keeps its class and attributes,
  and a ``Bspline`` pickled by the previous release (``main`` at 11ab436) still loads;
* ``evaluate`` and ``evaluate_derivatives`` return what ``out`` implies (a view of it, or it), fill
  a non-contiguous ``out`` through a temporary, and treat their points, orders and degenerate
  recurrences as documented.

Oracles are closed forms (the quadratic Bernstein curve ``2 t (1 - t)``, ``u v``) and the exact
rational oracle of :mod:`tests.oracles.bspline_exact`; the pickles of the previous release are
bytes written by that release's code.
"""

from __future__ import annotations

import base64
import copy
import pickle
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D
from pantr.quad import PointsLattice
from pantr.transform import AffineTransform
from tests.oracles import bspline_exact

_Array = npt.NDArray[np.float32 | np.float64]

_EPS64 = float(np.finfo(np.float64).eps)

# Bytes (protocol 4) written by the previous release (`main` at 11ab436) for
# a float32 curve (knots [0, 0, 0, .5, 1, 1, 1], degree 2, control points arange(4)) and for the
# degree-(2, 2) Bezier-like surface with control points `arange(18.0)` shaped 3 x 3 x 2.
_PICKLED_F32_CURVE = (
    "gASVYgIAAAAAAACMFnBhbnRyLmJzcGxpbmUuX2JzcGxpbmWUjAdCc3BsaW5llJOUKYGUfZQojAZfc3BhY2WUjB9w"
    "YW50ci5ic3BsaW5lLl9ic3BsaW5lX3NwYWNlX25klIwMQnNwbGluZVNwYWNllJOUKYGUfZQojAdfc3BhY2VzlIwf"
    "cGFudHIuYnNwbGluZS5fYnNwbGluZV9zcGFjZV8xZJSMDkJzcGxpbmVTcGFjZTFElJOUKYGUfZQojAZfa25vdHOU"
    "jBZudW1weS5fY29yZS5tdWx0aWFycmF5lIwMX3JlY29uc3RydWN0lJOUjAVudW1weZSMB25kYXJyYXmUk5RLAIWU"
    "QwFilIeUUpQoSwFLB4WUaBWMBWR0eXBllJOUjAJmNJSJiIeUUpQoSwOMATyUTk5OSv////9K/////0sAdJRiiUMc"
    "AAAAAAAAAAAAAAAAAAAAPwAAgD8AAIA/AACAP5R0lGKMBF90b2yURz6wAAAAAAAAjAdfZGVncmVllEsCjAlfcGVy"
    "aW9kaWOUiYwNbnVtX2ludGVydmFsc5RLAowJbnVtX2Jhc2lzlEsEdWKFlGgqSwSFlIwPbnVtX3RvdGFsX2Jhc2lz"
    "lEsEdWKMD19jb250cm9sX3BvaW50c5RoFGgXSwCFlGgZh5RSlChLAUsESwGGlGgejAJmNJSJiIeUUpQoSwNoIk5O"
    "Tkr/////Sv////9LAHSUYolDEAAAAAAAAIA/AAAAQAAAQECUdJRijAxfaXNfcmF0aW9uYWyUiYwOX2JlemllcnNf"
    "Y2FjaGWUTowNX2xvY2F0ZV9jYWNoZZROdWIu"
)
_PICKLED_SURFACE = (
    "gASVbgMAAAAAAACMFnBhbnRyLmJzcGxpbmUuX2JzcGxpbmWUjAdCc3BsaW5llJOUKYGUfZQojAZfc3BhY2WUjB9w"
    "YW50ci5ic3BsaW5lLl9ic3BsaW5lX3NwYWNlX25klIwMQnNwbGluZVNwYWNllJOUKYGUfZQojAdfc3BhY2VzlIwf"
    "cGFudHIuYnNwbGluZS5fYnNwbGluZV9zcGFjZV8xZJSMDkJzcGxpbmVTcGFjZTFElJOUKYGUfZQojAZfa25vdHOU"
    "jBZudW1weS5fY29yZS5tdWx0aWFycmF5lIwMX3JlY29uc3RydWN0lJOUjAVudW1weZSMB25kYXJyYXmUk5RLAIWU"
    "QwFilIeUUpQoSwFLBoWUaBWMBWR0eXBllJOUjAJmOJSJiIeUUpQoSwOMATyUTk5OSv////9K/////0sAdJRiiUMw"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA8D8AAAAAAADwPwAAAAAAAPA/lHSUYowEX3RvbJRHPOAAAAAA"
    "AACMB19kZWdyZWWUSwKMCV9wZXJpb2RpY5SJjA1udW1faW50ZXJ2YWxzlEsBjAludW1fYmFzaXOUSwN1YmgOKYGU"
    "fZQoaBFoFGgXSwCFlGgZh5RSlChLAUsGhZRoIYlDMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAPA/AAAA"
    "AAAA8D8AAAAAAADwP5R0lGJoJkc84AAAAAAAAGgnSwJoKIloKUsBaCpLA3VihpRoKksDSwOGlIwPbnVtX3RvdGFs"
    "X2Jhc2lzlEsJdWKMD19jb250cm9sX3BvaW50c5RoFGgXSwCFlGgZh5RSlChLAUsDSwNLAoeUaB6MAmY4lImIh5RS"
    "lChLA2giTk5OSv////9K/////0sAdJRiiUOQAAAAAAAAAAAAAAAAAADwPwAAAAAAAABAAAAAAAAACEAAAAAAAAAQ"
    "QAAAAAAAABRAAAAAAAAAGEAAAAAAAAAcQAAAAAAAACBAAAAAAAAAIkAAAAAAAAAkQAAAAAAAACZAAAAAAAAAKEAA"
    "AAAAAAAqQAAAAAAAACxAAAAAAAAALkAAAAAAAAAwQAAAAAAAADFAlHSUYowMX2lzX3JhdGlvbmFslImMDl9iZXpp"
    "ZXJzX2NhY2hllE6MDV9sb2NhdGVfY2FjaGWUTnViLg=="
)


def _space_1d(degree: int = 2, dtype: npt.DTypeLike = np.float64) -> BsplineSpace1D:
    """Build a Bezier-like 1D space of ``degree`` on [0, 1]."""
    knots = np.array([0.0] * (degree + 1) + [1.0] * (degree + 1), dtype=dtype)
    return BsplineSpace1D(knots, degree)


def _curve(dtype: npt.DTypeLike = np.float64) -> Bspline:
    """Build the quadratic curve with control points (0, 1, 0): ``f(t) = 2 t (1 - t)``."""
    return Bspline(BsplineSpace([_space_1d(2, dtype)]), np.array([0.0, 1.0, 0.0], dtype=dtype))


def _surface() -> Bspline:
    """Build the bilinear surface ``f(u, v) = u v`` (control points 0, 0, 0, 1)."""
    s = BsplineSpace([_space_1d(1), _space_1d(1)])
    return Bspline(s, np.array([0.0, 0.0, 0.0, 1.0]))


def _assert_read_only(arr: npt.NDArray[Any]) -> None:
    """Assert that ``arr`` is flagged read-only and refuses a write."""
    assert not arr.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        arr[...] = 0


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_constructor_copies_the_control_points_and_they_are_read_only() -> None:
    """A write to the caller's array does not reach the B-spline, and its points refuse writes."""
    cp = np.array([0.0, 1.0, 0.0])
    b = Bspline(BsplineSpace([_space_1d()]), cp)

    cp[1] = 5.0

    np.testing.assert_array_equal(b.control_points, [[0.0], [1.0], [0.0]])
    _assert_read_only(b.control_points)
    assert b.control_points is b.control_points  # read once, cached
    np.testing.assert_allclose(b.evaluate(np.array([0.25])), 0.375, rtol=0, atol=8 * _EPS64)


@pytest.mark.parametrize("read_first", [False, True], ids=["unread", "already-read"])
def test_control_points_refuse_assignment_and_deletion(read_first: bool) -> None:
    """``control_points`` is a read-only property, however it is reached."""
    b = _curve()
    if read_first:
        _ = b.control_points
    with pytest.raises(AttributeError, match="no setter"):
        b.control_points = np.zeros((3, 1))
    with pytest.raises(AttributeError, match="no deleter"):
        del b.control_points


def test_the_space_is_the_object_the_bspline_was_built_from() -> None:
    """``b.space is b.space``, and it is the caller's own object."""
    space = BsplineSpace([_space_1d()])
    b = Bspline(space, np.array([0.0, 1.0, 0.0]))
    assert b.space is space
    assert b.space is b.space


def test_a_one_dimensional_masked_list_and_zero_dimensional_input_are_data() -> None:
    """Masked arrays contribute their data, lists convert, a 0-d input fills one function."""
    space = BsplineSpace([_space_1d()])
    masked = np.ma.masked_array([0.0, 1.0, 0.0], mask=np.array([False, True, False]))
    np.testing.assert_array_equal(Bspline(space, masked).control_points[:, 0], [0.0, 1.0, 0.0])
    np.testing.assert_array_equal(Bspline(space, [0.0, 1.0, 0.0]).control_points[:, 0], [0, 1, 0])
    one = BsplineSpace([BsplineSpace1D(np.array([0.0, 1.0]), 0)])
    assert Bspline(one, np.array(3.0)).control_points.shape == (1, 1)


def test_control_points_of_another_layout_are_copied_c_contiguous() -> None:
    """A Fortran-ordered input is copied into a C-contiguous array of the same values."""
    space = BsplineSpace([_space_1d()])
    cp = np.asfortranarray(np.arange(6.0).reshape(3, 2))
    b = Bspline(space, cp)
    assert b.control_points.flags.c_contiguous
    np.testing.assert_array_equal(b.control_points, cp)


@pytest.mark.parametrize(
    ("cp", "pattern"),
    [
        pytest.param(
            np.zeros(4), "multiple of the number of basis functions.Got 4 control", id="count"
        ),
        pytest.param(
            np.zeros(3, dtype=np.float32),
            "same dtype as the B-spline space.Got float32",
            id="dtype",
        ),
        pytest.param(np.zeros(3, dtype=np.int64), "Got int64 control points and float64", id="int"),
        pytest.param(np.zeros(3, dtype=np.float16), "Got float16 control points", id="float16"),
        pytest.param(
            np.zeros(3, dtype=np.longdouble), "float128|float96|longdouble", id="longdouble"
        ),
        pytest.param(np.zeros(3, dtype=bool), "Got bool control points", id="bool"),
        pytest.param(np.zeros(3, dtype=">f8"), "Got >f8 control points", id="byte-swapped"),
        pytest.param(np.zeros(3, dtype=complex), "Got complex128 control points", id="complex"),
        pytest.param(np.zeros(0), "at least rank one. Got rank 0$", id="rank-zero"),
    ],
)
def test_construction_refusals_keep_the_messages_of_the_previous_release(
    cp: _Array, pattern: str
) -> None:
    """The count, dtype and rank checks raise ``ValueError`` with the wording the suite matched."""
    with pytest.raises(ValueError, match=pattern):
        Bspline(BsplineSpace([_space_1d()]), cp)


def test_a_rational_spline_needs_a_weight_column() -> None:
    """One coordinate per point leaves rank 0 when rational."""
    with pytest.raises(ValueError, match="at least rank one. Got rank 0$"):
        Bspline(BsplineSpace([_space_1d()]), np.zeros(3), is_rational=True)


def test_the_checks_run_in_the_order_count_dtype_rank() -> None:
    """With two things wrong, the first of count, dtype, rank is reported."""
    f32 = BsplineSpace([_space_1d(2, np.float32)])
    with pytest.raises(ValueError, match="multiple of the number of basis functions"):
        Bspline(f32, np.zeros(4))  # wrong count and wrong dtype
    with pytest.raises(ValueError, match="same dtype"):
        Bspline(f32, np.zeros(3), is_rational=True)  # wrong dtype and rank 0


def test_a_space_without_a_direction_raises_index_error_as_before() -> None:
    """The empty space has no dtype: ``IndexError``, the error of the previous release."""
    with pytest.raises(IndexError):
        Bspline(BsplineSpace([]), np.zeros(3))


# ---------------------------------------------------------------------------
# In-place operations replace the value
# ---------------------------------------------------------------------------


def _surface_with_distinct_points() -> Bspline:
    """Build a degree-(1, 2) Bezier-like surface of rank 2, all entries distinct."""
    s = BsplineSpace([_space_1d(1), _space_1d(2)])
    return Bspline(s, np.arange(12.0).reshape(2, 3, 2) + 1.0)


_MOVES: dict[str, Callable[[Bspline], None]] = {
    "reverse": lambda b: b.reverse(1, in_place=True),
    "permute": lambda b: b.permute_directions([1, 0], in_place=True),
    "transform": lambda b: b.transform(AffineTransform.translation([1.0, 2.0]), in_place=True),
}


@pytest.mark.parametrize("move", list(_MOVES))
def test_in_place_move_replaces_the_value_and_keeps_an_earlier_view(move: str) -> None:
    """The array read before keeps its values; the object shows the new ones; caches reset."""
    b = _surface_with_distinct_points()
    before = b.control_points
    snapshot = before.copy()
    old_space = b.space
    b.to_beziers()  # fill the cache
    assert getattr(b, "_beziers_cache") is not None  # noqa: B009

    _MOVES[move](b)

    np.testing.assert_array_equal(before, snapshot)
    assert b.control_points is not before
    _assert_read_only(b.control_points)
    assert getattr(b, "_beziers_cache") is None  # noqa: B009
    assert getattr(b, "_locate_cache") is None  # noqa: B009
    if move == "transform":
        assert b.space is old_space
        np.testing.assert_array_equal(b.control_points, snapshot + np.array([1.0, 2.0]))
    elif move == "reverse":
        np.testing.assert_array_equal(b.control_points, np.flip(snapshot, axis=1))
        assert b.space is not old_space
    else:
        np.testing.assert_array_equal(b.control_points, snapshot.transpose(1, 0, 2))
        assert b.degree == (2, 1)


@pytest.mark.parametrize("move", ["reverse", "permute", "transform"])
def test_a_non_in_place_move_is_independent_of_the_original(move: str) -> None:
    """The returned B-spline owns its points: nothing it does reaches the original."""
    b = _surface_with_distinct_points()
    snapshot = b.control_points.copy()
    result = {
        "reverse": lambda: b.reverse(0),
        "permute": lambda: b.permute_directions([1, 0]),
        "transform": lambda: b.transform(AffineTransform.translation([1.0, 2.0])),
    }[move]()
    assert result is not b
    assert not np.shares_memory(result.control_points, b.control_points)
    _assert_read_only(result.control_points)
    np.testing.assert_array_equal(b.control_points, snapshot)


# ---------------------------------------------------------------------------
# copy, deepcopy, pickle
# ---------------------------------------------------------------------------


def _pickled(protocol: int) -> Callable[[Any], Any]:
    """Return a round trip through ``pickle`` at ``protocol``."""
    return lambda x: pickle.loads(pickle.dumps(x, protocol=protocol))


_PICKLES = [pytest.param(_pickled(p), id=f"pickle-{p}") for p in range(pickle.HIGHEST_PROTOCOL + 1)]
_DEEP_COPIERS = [pytest.param(copy.deepcopy, id="deepcopy"), *_PICKLES]
_COPIERS = [pytest.param(copy.copy, id="copy"), *_DEEP_COPIERS]


class _TaggedBspline(Bspline):
    """A subclass with attributes of its own, which copies and pickles must carry."""

    label: str
    history: list[Any]


@pytest.mark.parametrize("copier", _COPIERS)
@pytest.mark.parametrize(
    ("make", "rational"),
    [
        pytest.param(_curve, False, id="curve-f64"),
        pytest.param(lambda: _curve(np.float32), False, id="curve-f32"),
        pytest.param(_surface, False, id="surface"),
        pytest.param(_surface_with_distinct_points, False, id="vector-surface"),
    ],
)
def test_round_trip_is_equal_by_value_and_read_only(
    copier: Callable[[Bspline], Bspline], make: Callable[[], Bspline], rational: bool
) -> None:
    """A copy equals the original by value, in dtype and space too, and its points are read-only."""
    original = make()

    got = copier(original)

    assert type(got) is Bspline
    assert got.is_rational is rational
    assert got.dtype == original.dtype
    assert got.degree == original.degree
    assert got.rank == original.rank
    assert got.dim == original.dim
    np.testing.assert_array_equal(got.control_points, original.control_points)
    assert got.control_points.dtype == original.control_points.dtype
    _assert_read_only(got.control_points)
    pts = np.full((2, original.dim), 0.3, dtype=original.dtype)
    if original.dim == 1:
        pts = pts[:, 0]
    np.testing.assert_array_equal(got.evaluate(pts), original.evaluate(pts))


@pytest.mark.parametrize("copier", _COPIERS)
def test_a_rational_round_trip_keeps_the_weights(copier: Callable[[Bspline], Bspline]) -> None:
    """The flag and the weight column survive, and the rank excludes the weight."""
    cp = np.array([[0.0, 1.0], [1.0, 2.0], [0.0, 1.0]])
    original = Bspline(BsplineSpace([_space_1d()]), cp, is_rational=True)

    got = copier(original)

    assert got.is_rational is True
    assert got.rank == 1
    np.testing.assert_array_equal(got.control_points, cp)
    t = np.array([0.0, 0.3, 1.0])
    np.testing.assert_array_equal(got.evaluate(t), original.evaluate(t))


@pytest.mark.parametrize("copier", _COPIERS)
def test_in_place_change_of_a_copy_leaves_the_original(
    copier: Callable[[Bspline], Bspline],
) -> None:
    """``copy.copy`` shares the compiled value, so ``in_place`` must replace, not write into it."""
    original = _surface_with_distinct_points()
    snapshot = original.control_points.copy()
    twin = copier(original)

    twin.reverse(0, in_place=True)

    np.testing.assert_array_equal(original.control_points, snapshot)
    np.testing.assert_array_equal(twin.control_points, np.flip(snapshot, axis=0))


@pytest.mark.parametrize("copier", _COPIERS)
@pytest.mark.parametrize("rational", [False, True], ids=["polynomial", "rational"])
def test_subclass_keeps_its_class_and_attributes(
    copier: Callable[[Bspline], Bspline], rational: bool
) -> None:
    """A subclass with attributes of its own survives copy, deepcopy and pickle as itself."""
    cp = np.array([[0.0, 1.0], [1.0, 2.0], [0.0, 1.0]], dtype=np.float32)
    space = BsplineSpace([_space_1d(2, np.float32)])
    original = _TaggedBspline(space, cp if rational else cp[:, :1], is_rational=rational)
    original.label = "left-wall"
    original.history = [1, [2, 3]]
    _ = original.control_points  # the cached view is derived, not state

    got = copier(original)

    assert type(got) is _TaggedBspline
    assert got.label == "left-wall"
    assert got.history == [1, [2, 3]]
    assert got.is_rational is rational
    assert got.dtype == np.float32


@pytest.mark.parametrize("copier", _DEEP_COPIERS)
def test_deep_copy_does_not_share_a_mutable_attribute(copier: Callable[[Bspline], Bspline]) -> None:
    """Deep means the subclass's own attributes are copied too."""
    original = _TaggedBspline(BsplineSpace([_space_1d()]), np.array([0.0, 1.0, 0.0]))
    original.history = [1, [2, 3]]

    got = copier(original)

    assert isinstance(got, _TaggedBspline)
    assert got.history == original.history
    assert got.history is not original.history
    got.history[1].append(4)
    assert original.history == [1, [2, 3]]


def test_a_located_spline_pickles() -> None:
    """The point-inversion cache is derived state: it is not pickled (it could not be before)."""
    s = BsplineSpace([_space_1d(1), _space_1d(1)])
    b = Bspline(s, np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]))
    b.locate(np.array([[0.5, 0.5]]))
    assert b._locate_cache is not None

    got = pickle.loads(pickle.dumps(b))

    assert got._locate_cache is None
    np.testing.assert_array_equal(got.control_points, b.control_points)


# ---------------------------------------------------------------------------
# Pickles written by the previous release
# ---------------------------------------------------------------------------


def _plain(cls: type, **attrs: object) -> object:
    """Build an instance of ``cls`` with exactly the attributes ``attrs``, bypassing any init."""
    obj = cls()
    obj.__dict__.update(attrs)
    return obj


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_a_bspline_pickled_by_the_previous_release_loads_and_evaluates(protocol: int) -> None:
    """The old plain attributes (``_space``, ``_control_points``, the caches) rebuild the value."""
    import pantr.bspline._bspline as bspline_module  # noqa: PLC0415

    space = BsplineSpace([_space_1d(2, np.float32)])
    cp = np.array([[0.0], [1.0], [0.0]], dtype=np.float32)
    stand_in = type("Bspline", (), {"__module__": bspline_module.__name__})
    obj = _plain(
        stand_in,
        _space=space,
        _control_points=cp.copy(),
        _is_rational=False,
        _beziers_cache=None,
        _locate_cache=None,
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(bspline_module, "Bspline", stand_in)
        data = pickle.dumps(obj, protocol=protocol)

    loaded = pickle.loads(data)

    assert type(loaded) is Bspline
    assert loaded.dtype == np.float32
    assert loaded.is_rational is False
    np.testing.assert_array_equal(loaded.control_points, cp)
    _assert_read_only(loaded.control_points)
    np.testing.assert_allclose(
        loaded.evaluate(np.array([0.25], dtype=np.float32)), 0.375, rtol=0, atol=8 * 2**-23
    )
    loaded.reverse(in_place=True)  # the loaded object is a full value, not just readable
    np.testing.assert_array_equal(loaded.control_points, cp[::-1])


def test_bytes_written_by_the_previous_release_load() -> None:
    """Two real pickles of ``main`` 11ab436 (a float32 curve, a surface) load and evaluate.

    The bytes carry the old plain attributes and an unset Bézier cache. The oracle for the values
    is the exact rational oracle on the knots and points the bytes were written from.
    """
    curve = pickle.loads(base64.b64decode(_PICKLED_F32_CURVE))
    assert type(curve) is Bspline
    assert curve.dtype == np.float32
    assert curve.degree == (2,)
    np.testing.assert_array_equal(curve.control_points[:, 0], np.arange(4, dtype=np.float32))
    knots = [np.array([0, 0, 0, 0.5, 1, 1, 1.0])]
    t = np.array([0.1, 0.3, 0.75], dtype=np.float32)
    got = curve.evaluate(t)
    for x, g in zip(t, got, strict=True):
        want = bspline_exact.derivative(knots, [2], np.arange(4.0).reshape(4, 1), [float(x)])
        assert bspline_exact.within(g, want[0], 16 * 2**-23)

    surface = pickle.loads(base64.b64decode(_PICKLED_SURFACE))
    assert type(surface) is Bspline
    assert surface.degree == (2, 2)
    assert surface.control_points.shape == (3, 3, 2)
    np.testing.assert_array_equal(surface.control_points, np.arange(18.0).reshape(3, 3, 2))
    assert surface._beziers_cache is None


# ---------------------------------------------------------------------------
# Evaluation: what comes back, and what is refused
# ---------------------------------------------------------------------------


def test_evaluate_returns_a_squeezed_view_of_out() -> None:
    """With ``out`` the result is a view of it (not ``out`` itself), squeezed as without."""
    b = _curve()
    out = np.empty((2, 1))

    got = b.evaluate(np.array([0.25, 0.5]), out)

    assert got.shape == (2,)
    assert np.shares_memory(got, out)
    assert got is not out
    np.testing.assert_allclose(out[:, 0], [0.375, 0.5], rtol=0, atol=8 * _EPS64)


def test_evaluate_leaves_the_weight_column_of_a_rational_out_and_returns_the_rest() -> None:
    """``out`` has the weight column (divided coordinates and the weight); the result drops it."""
    cp = np.array([[0.0, 1.0], [1.0, 2.0], [0.0, 1.0]])
    b = Bspline(BsplineSpace([_space_1d()]), cp, is_rational=True)
    out = np.empty((2, 2))

    got = b.evaluate(np.array([0.25, 0.5]), out)

    assert got.shape == (2,)
    np.testing.assert_array_equal(got, out[:, 0])
    # w(t) = (1 - t)^2 + 4 t (1 - t) + t^2 and x(t) = 2 t (1 - t), so x / w.
    t = np.array([0.25, 0.5])
    w = (1 - t) ** 2 + 4 * t * (1 - t) + t**2
    np.testing.assert_allclose(out[:, 1], w, rtol=0, atol=8 * _EPS64)
    np.testing.assert_allclose(out[:, 0], 2 * t * (1 - t) / w, rtol=0, atol=16 * _EPS64)


def test_a_non_contiguous_out_is_filled() -> None:
    """A strided or Fortran-ordered ``out`` gets the values, as before."""
    b = _surface()
    big = np.zeros((2, 2))
    b.evaluate(np.array([[0.5, 0.5], [1.0, 1.0]]), big[:, :1])
    np.testing.assert_allclose(big, [[0.25, 0.0], [1.0, 0.0]], rtol=0, atol=8 * _EPS64)
    f_out = np.empty((2, 3, 1), order="F")
    lattice = PointsLattice([np.array([0.0, 1.0]), np.array([0.0, 0.5, 1.0])])
    got = b.evaluate(lattice, f_out)
    np.testing.assert_allclose(
        f_out[..., 0], np.array([[0, 0, 0], [0, 0.5, 1]]), rtol=0, atol=8 * _EPS64
    )
    assert np.shares_memory(got, f_out)


@pytest.mark.parametrize(
    ("make_out", "pattern"),
    [
        pytest.param(lambda: np.empty(2), r"shape \(2,\), but expected shape \(2, 1\)", id="shape"),
        pytest.param(lambda: np.empty((2, 1), np.float32), "dtype float32", id="dtype"),
    ],
)
def test_out_of_the_wrong_shape_or_dtype_is_refused(
    make_out: Callable[[], _Array], pattern: str
) -> None:
    """The ``out`` of evaluate has the shape ``(n, cp_size)``, even for a scalar field."""
    with pytest.raises(ValueError, match=pattern):
        _curve().evaluate(np.array([0.25, 0.5]), make_out())


def test_a_read_only_out_is_refused() -> None:
    """``out`` must be writeable."""
    out = np.empty((2, 1))
    out.flags.writeable = False
    with pytest.raises(ValueError, match="not writeable"):
        _curve().evaluate(np.array([0.25, 0.5]), out)


def test_evaluate_derivatives_fills_and_returns_out_itself() -> None:
    """The derivative takes ``out`` of the result's shape and returns that very array."""
    b = _surface()
    out = np.empty(2)

    got = b.evaluate_derivatives(np.array([[0.25, 0.5], [0.1, 0.2]]), [1, 0], out)

    assert got is out
    np.testing.assert_allclose(out, [0.5, 0.2], rtol=0, atol=8 * _EPS64)  # d(uv)/du = v
    with pytest.raises(ValueError, match="shape"):
        b.evaluate_derivatives(np.array([[0.25, 0.5]]), [1, 0], np.empty(3))
    f = np.zeros((2, 2))
    b.evaluate_derivatives(np.array([[0.25, 0.5], [0.1, 0.2]]), [1, 0], f[:, 0])
    np.testing.assert_allclose(f, [[0.5, 0.0], [0.2, 0.0]], rtol=0, atol=8 * _EPS64)


@pytest.mark.parametrize(
    "orders",
    [
        pytest.param([np.int64(1)], id="numpy-int"),
        pytest.param(np.array([1], dtype=np.uint8), id="uint8-array"),
        pytest.param((1,), id="tuple"),
        pytest.param([True], id="bool"),
    ],
)
def test_orders_may_be_any_sequence_of_integers(orders: Any) -> None:
    """Any integer type is an order; the first derivative of ``2 t (1 - t)`` is ``2 - 4 t``."""
    got = _curve().evaluate_derivatives(np.array([0.25]), orders)
    np.testing.assert_allclose(got, [1.0], rtol=0, atol=8 * _EPS64)


class _Index:
    """An object that is an integer only by ``__index__``."""

    def __init__(self, value: int) -> None:
        self.value = value

    def __index__(self) -> int:
        return self.value


def test_an_order_with_dunder_index_is_accepted_and_a_float_is_not() -> None:
    """``operator.index`` is the rule: ``__index__`` objects pass, floats are a ``TypeError``."""
    b = _curve()
    index_order: Any = [_Index(1)]
    got = b.evaluate_derivatives(np.array([0.25]), index_order)
    np.testing.assert_allclose(got, [1.0], rtol=0, atol=8 * _EPS64)
    with pytest.raises(TypeError):
        b.evaluate_derivatives(np.array([0.25]), cast(Any, [1.0]))
    with pytest.raises(TypeError):
        b.evaluate_derivatives(np.array([0.25]), 1.0)  # type: ignore[arg-type]


def test_a_single_order_is_broadcast_to_every_direction() -> None:
    """An integer ``orders`` is that order in each direction: ``d2(u v) / du dv = 1``."""
    got = _surface().evaluate_derivatives(np.array([[0.3, 0.4]]), 1)
    np.testing.assert_allclose(got, [1.0], rtol=0, atol=8 * _EPS64)


@pytest.mark.parametrize(
    ("orders", "pattern"),
    [
        pytest.param([1, 0], r"len\(orders\) must equal spline.dim \(1\), got 2", id="too-many"),
        pytest.param([], r"len\(orders\) must equal spline.dim \(1\), got 0", id="none"),
        pytest.param([-1], r"orders\[0\] must be >= 0, got -1", id="negative"),
    ],
)
def test_orders_of_the_wrong_number_or_sign_are_refused(orders: list[int], pattern: str) -> None:
    """A 1D B-spline takes exactly one order (it used to ignore extra ones)."""
    with pytest.raises(ValueError, match=pattern):
        _curve().evaluate_derivatives(np.array([0.25]), orders)


def test_an_order_beyond_int_is_refused_and_one_above_the_degree_is_zero() -> None:
    """A huge order is a ``ValueError`` (the Numba kernel ran out of memory); above degree: 0."""
    b = _curve()
    with pytest.raises(ValueError, match="too large"):
        b.evaluate_derivatives(np.array([0.25]), [2**40])
    np.testing.assert_array_equal(b.evaluate_derivatives(np.array([0.25]), [3]), [0.0])
    np.testing.assert_array_equal(b.evaluate_derivatives(np.array([0.25]), [100]), [0.0])


_LEFT_HEAVY = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
"""Degree 2 with the left end repeated four times (p + 2): the span just left of 0 is empty."""


def test_a_zero_divisor_in_the_derivative_recurrence_raises_in_every_dimension() -> None:
    """A point inside the tolerance band of the empty span raises ``ZeroDivisionError``.

    The 1D evaluation returned ``nan`` there before, silently; every dimension now raises, as the
    n-dimensional evaluation and the basis tabulation did.
    """
    s1 = BsplineSpace([BsplineSpace1D(_LEFT_HEAVY, 2)])
    b1 = Bspline(s1, np.arange(4.0))
    bad = np.array([-1e-17])
    with pytest.raises(ZeroDivisionError):
        b1.evaluate_derivatives(bad, [1])
    s2 = BsplineSpace([BsplineSpace1D(_LEFT_HEAVY, 2), _space_1d(1)])
    with pytest.raises(ZeroDivisionError):
        Bspline(s2, np.arange(8.0)).evaluate_derivatives(np.array([[-1e-17, 0.5]]), [1, 0])
    rational = Bspline(s1, np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0]]), True)
    with pytest.raises(ZeroDivisionError):
        rational.evaluate_derivatives(bad, [1])
    # Order zero divides by nothing, and the value is finite.
    assert np.isfinite(b1.evaluate_derivatives(bad, [0])).all()
    assert np.isfinite(b1.evaluate(bad)).all()


def test_points_outside_the_domain_and_nan_are_refused() -> None:
    """Evaluation names the domain; a NaN point is outside it."""
    b = _curve()
    with pytest.raises(ValueError, match="outside the knot vector domain"):
        b.evaluate(np.array([1.5]))
    with pytest.raises(ValueError, match="outside the knot vector domain"):
        b.evaluate(np.array([np.nan]))
    with pytest.raises(ValueError, match="outside the knot vector domain"):
        _surface().evaluate_derivatives(np.array([[0.2, 1.5]]), [1, 0])


def test_a_one_dimensional_lattice_may_have_another_float_dtype_than_the_spline() -> None:
    """Only the 1D evaluation takes a lattice of the other float dtype, as before.

    The values are computed in the common type of the knots and the points and returned in the
    spline's dtype; a lattice of two directions must match the dtype.
    """
    b32 = _curve(np.float32)
    t = np.array([0.1, 0.3, 0.7123456789012345])
    got = b32.evaluate(PointsLattice([t]))
    assert got.dtype == np.float32
    want = 2 * t * (1 - t)
    np.testing.assert_allclose(got, want, rtol=0, atol=16 * 2**-23)
    b64 = _curve()
    got64 = b64.evaluate(PointsLattice([t.astype(np.float32)]))
    assert got64.dtype == np.float64
    surface = _surface()
    with pytest.raises(ValueError, match="Points dtype must match B-spline dtype"):
        surface.evaluate(PointsLattice([np.array([0.5], np.float32), np.array([0.5], np.float32)]))
    with pytest.raises(ValueError, match="Points dtype must match B-spline dtype"):
        surface.evaluate(np.array([[0.5, 0.5]], dtype=np.float32))
    with pytest.raises(ValueError, match="Points lattice must be 1D"):
        b64.evaluate(PointsLattice([np.array([0.5]), np.array([0.5])]))
    with pytest.raises(ValueError, match="does not match B-spline dimension 2"):
        surface.evaluate(PointsLattice([np.array([0.5])]))


@pytest.mark.parametrize(
    ("pts", "pattern"),
    [
        pytest.param(np.array(0.25), r"shape \(n_pts,\) or \(n_pts, 1\).*got shape \(\)", id="0d"),
        pytest.param(np.array([1, 0]), "Points dtype must match B-spline dtype", id="int"),
        pytest.param(np.array([0.25], np.float16), "Points dtype must match", id="float16"),
        pytest.param(np.array([0.25]).astype(">f8"), "Points dtype must match", id="byte-swapped"),
        pytest.param(np.array([True, False]), "Points dtype must match", id="bool"),
    ],
)
def test_points_of_another_shape_or_dtype_are_refused(pts: Any, pattern: str) -> None:
    """The point checks keep their messages (the normalisation stays in Python)."""
    with pytest.raises(ValueError, match=pattern):
        _curve().evaluate(pts)


def test_points_given_as_a_masked_array_or_a_list_are_their_data() -> None:
    """Masked arrays contribute their data; lists convert."""
    b = _curve()
    masked = np.ma.masked_array([0.25, 0.5], mask=np.array([False, True]))
    np.testing.assert_allclose(b.evaluate(masked), [0.375, 0.5], rtol=0, atol=8 * _EPS64)
    np.testing.assert_allclose(b.evaluate([0.25, 0.5]), [0.375, 0.5], rtol=0, atol=8 * _EPS64)
    np.testing.assert_allclose(_surface().evaluate([[0.5, 0.5]]), 0.25, rtol=0, atol=8 * _EPS64)
