"""The root-provider switch of :mod:`pantr.bezier` and the incompleteness warning.

Covers design section 4.1 and ``design/root_seam.md`` from Python: the pinned wire values and the
Builtin provider's declared capabilities, the initial provider (``PANTR_ROOT_PROVIDER``, then the
installed default), ``set_root_provider`` / ``get_root_provider`` and their errors, the routing of
every public entry point to the active provider and back, and the ``RuntimeWarning`` a result with a
gap raises.

Oracles: the wire values are a closed set of integers fixed by the design; the capabilities are the
ones the design states for ``Builtin``; the routing is checked against a recording fake of the
``pantr_berf`` module (the provider module is the seam, so a fake there mocks nothing inside the
logic under test); the incompleteness case is a linear polynomial elevated to degree 8, whose root
``0.13881295865011115`` is analytic, solved with a tolerance below the spacing of doubles so that
the real depth cap is reached.
"""

from __future__ import annotations

import sys
import types
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from pantr._ext import ext
from pantr._testing import (
    find_roots_detailed,
    root_provider_capabilities,
    root_provider_state,
    set_root_provider_state,
)
from pantr.bezier import (
    Bezier,
    find_monotone_root,
    find_roots,
    get_root_provider,
    set_root_provider,
)
from pantr.tolerance import get_strict

_ENV = "PANTR_ROOT_PROVIDER"

# A linear polynomial elevated to degree 8 (coefficients of t - 0.13881295865011115 in the Bernstein
# basis) and a tolerance below the spacing of doubles at its root: no interval can narrow to it, so
# Bezier clipping reaches its depth cap.
_CAP_COEFFS = np.array(
    [
        -0.13881295865011115,
        -0.013812958650111146,
        0.11118704134988885,
        0.23618704134988885,
        0.3611870413498889,
        0.4861870413498889,
        0.6111870413498889,
        0.7361870413498889,
        0.8611870413498889,
    ]
)
_CAP_TOL = 1.2388055965619009e-52
_CAP_ROOT = 0.13881295865011115
_EPS = float(np.finfo(np.float64).eps)

_REQUIRED_FUNCTIONS = (
    "find_roots",
    "find_roots_batch",
    "find_monotone_root",
    "find_monotone_root_batch",
)


class _FakeBerf(types.ModuleType):
    """A stand-in for ``pantr_berf``: records its calls, answers with canned results."""

    def __init__(self, completeness: int | list[int] = 0) -> None:
        super().__init__("pantr_berf")
        self.completeness = completeness
        self.calls: list[tuple[str, np.dtype[Any], tuple[int, ...], bool, tuple[float, ...]]] = []

    def _record(self, name: str, coeff: npt.NDArray[Any], *tols: float) -> None:
        self.calls.append((name, coeff.dtype, coeff.shape, bool(coeff.flags.c_contiguous), tols))

    def _per_row(self, n: int) -> npt.NDArray[np.int32]:
        if isinstance(self.completeness, list):
            return np.asarray(self.completeness, dtype=np.int32)
        return np.full(n, self.completeness, dtype=np.int32)

    def _single(self, name: str, coeff: npt.NDArray[Any], pt: float, vt: float) -> Any:
        self._record(name, coeff, pt, vt)
        completeness = self.completeness if isinstance(self.completeness, int) else 0
        return np.array([0.25, 0.75]), completeness

    def _batch(self, name: str, coeffs: npt.NDArray[Any], pt: float, vt: float) -> Any:
        self._record(name, coeffs, pt, vt)
        n, n_coeff = coeffs.shape
        roots = np.full((n, max(n_coeff - 1, 1)), np.nan)
        roots[:, 0] = 0.25
        return roots, np.ones(n, dtype=np.int64), self._per_row(n)

    def find_roots(self, coeff: npt.NDArray[Any], pt: float, vt: float) -> Any:
        return self._single("find_roots", coeff, pt, vt)

    def find_roots_f32(self, coeff: npt.NDArray[Any], pt: float, vt: float) -> Any:
        return self._single("find_roots_f32", coeff, pt, vt)

    def find_roots_batch(self, coeffs: npt.NDArray[Any], pt: float, vt: float) -> Any:
        return self._batch("find_roots_batch", coeffs, pt, vt)

    def find_roots_batch_f32(
        self,
        coeffs: npt.NDArray[Any],
        pt: float,
        vt: float,
    ) -> Any:
        return self._batch("find_roots_batch_f32", coeffs, pt, vt)

    def find_monotone_root(self, coeff: npt.NDArray[Any], pt: float) -> float:
        self._record("find_monotone_root", coeff, pt)
        return 0.5

    def find_monotone_root_f32(self, coeff: npt.NDArray[Any], pt: float) -> float:
        self._record("find_monotone_root_f32", coeff, pt)
        return 0.5

    def find_monotone_root_batch(
        self, coeffs: npt.NDArray[Any], pt: float
    ) -> npt.NDArray[np.float64]:
        self._record("find_monotone_root_batch", coeffs, pt)
        return np.full(coeffs.shape[0], 0.5)

    def find_monotone_root_batch_f32(
        self, coeffs: npt.NDArray[Any], pt: float
    ) -> npt.NDArray[np.float64]:
        self._record("find_monotone_root_batch_f32", coeffs, pt)
        return np.full(coeffs.shape[0], 0.5)


@pytest.fixture(autouse=True)
def _isolated_provider(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Run each test with the provider state restored afterwards and no ambient provider.

    The provider state is saved and restored; the environment variable is removed; a real
    ``pantr_berf`` is hidden (a ``None`` entry in ``sys.modules`` makes its import fail), so no
    test depends on whether it is installed.

    Yields:
        None
    """
    saved = root_provider_state()
    monkeypatch.delenv(_ENV, raising=False)
    monkeypatch.setitem(sys.modules, "pantr_berf", None)
    yield
    set_root_provider_state(saved)


def _reset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return the provider state to "not yet resolved" (the fixture restores it)."""
    del monkeypatch  # the callers pass it; the fixture does the restoring
    set_root_provider_state(None)


def _install_fake(monkeypatch: pytest.MonkeyPatch, fake: _FakeBerf) -> None:
    monkeypatch.setitem(sys.modules, "pantr_berf", fake)


def _line() -> Bezier:
    """A degree-1 Bezier with the root 0.5 (the Builtin's answer, unlike the fake's)."""
    return Bezier(np.array([1.0, -1.0]))


# --- The wire format and the Builtin provider ---------------------------------------------------


def test_wire_values_are_pinned() -> None:
    """The claim and completeness codes cross into Python and between modules: they never change."""
    assert (ext.roots.CLAIM_NONE, ext.roots.CLAIM_COMPUTED, ext.roots.CLAIM_CERTIFIED) == (0, 1, 2)
    assert (ext.roots.COMPLETE, ext.roots.GAP_LOCALIZED, ext.roots.GAP_UNLOCALIZED) == (0, 1, 2)


def test_builtin_states_its_capabilities() -> None:
    """The Builtin provider claims ``computed``, has batch and monotone solvers, no multiplicity."""
    max_claim, batch, monotone, multiplicity = root_provider_capabilities()
    assert max_claim == ext.roots.CLAIM_COMPUTED
    assert (batch, monotone, multiplicity) == (True, True, False)


def test_builtin_claims_never_exceed_its_maximum() -> None:
    """Every root is ``computed`` with multiplicity -1, and an ordinary result is complete."""
    # (t - 1/4)(t - 3/4) in the Bernstein basis: (3/16, -5/16, 3/16).
    coeff = np.array([3.0 / 16.0, -5.0 / 16.0, 3.0 / 16.0])
    roots, claims, multiplicities, completeness, gaps = find_roots_detailed(coeff, 1e-12, 1e-12)
    np.testing.assert_allclose(roots, [0.25, 0.75], atol=1e-11)
    assert claims.tolist() == [ext.roots.CLAIM_COMPUTED] * 2
    assert int(max(claims)) <= root_provider_capabilities()[0]
    assert multiplicities.tolist() == [-1, -1]
    assert completeness == ext.roots.COMPLETE
    assert gaps.shape == (0, 2)


def test_builtin_exposes_the_interface_a_provider_module_needs() -> None:
    """The compiled Builtin has every function the wrapper asks of a provider, in both dtypes."""
    for name in _REQUIRED_FUNCTIONS:
        assert callable(getattr(ext.roots, name))
        assert callable(getattr(ext.roots, name + "_f32"))


# --- The initial provider -------------------------------------------------------------------------


def test_default_provider_is_builtin_without_pantr_berf(monkeypatch: pytest.MonkeyPatch) -> None:
    """With nothing set and ``pantr_berf`` absent, the first use resolves to ``builtin``."""
    _reset(monkeypatch)
    assert get_root_provider() == "builtin"


def test_default_provider_is_berf_when_pantr_berf_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    """With nothing set and ``pantr_berf`` importable, the first use resolves to ``berf``."""
    _install_fake(monkeypatch, _FakeBerf())
    _reset(monkeypatch)
    assert get_root_provider() == "berf"


def test_environment_unknown_name_raises_on_first_use(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown ``PANTR_ROOT_PROVIDER`` raises at first use, every time, never at import."""
    monkeypatch.setenv(_ENV, "gurobi")
    _reset(monkeypatch)
    with pytest.raises(ValueError, match="PANTR_ROOT_PROVIDER"):
        get_root_provider()
    # Nothing was cached: the failure repeats, through the public entry points too.
    with pytest.raises(ValueError, match="PANTR_ROOT_PROVIDER"):
        find_roots(_line())
    with pytest.raises(ValueError, match="PANTR_ROOT_PROVIDER"):
        find_monotone_root(_line())
    # An explicit choice replaces the environment's, which is not consulted again.
    set_root_provider("builtin")
    assert get_root_provider() == "builtin"


def test_environment_builtin_is_honoured_over_an_importable_berf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``PANTR_ROOT_PROVIDER=builtin`` wins over the "berf when installed" default."""
    fake = _FakeBerf()
    _install_fake(monkeypatch, fake)
    monkeypatch.setenv(_ENV, "builtin")
    _reset(monkeypatch)
    assert get_root_provider() == "builtin"
    np.testing.assert_allclose(find_roots(_line()), [0.5], atol=1e-12)
    assert fake.calls == []


def test_environment_berf_is_honoured_and_needs_the_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``PANTR_ROOT_PROVIDER=berf`` selects it when importable, and fails loudly when not."""
    monkeypatch.setenv(_ENV, "berf")
    _reset(monkeypatch)
    with pytest.raises(ImportError, match="pantr-berf"):
        get_root_provider()
    _install_fake(monkeypatch, _FakeBerf())
    assert get_root_provider() == "berf"


def test_empty_environment_value_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty ``PANTR_ROOT_PROVIDER`` is the same as unset."""
    monkeypatch.setenv(_ENV, "")
    _reset(monkeypatch)
    assert get_root_provider() == "builtin"


def test_environment_is_read_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Changing the variable after first use changes nothing."""
    monkeypatch.setenv(_ENV, "builtin")
    _reset(monkeypatch)
    assert get_root_provider() == "builtin"
    monkeypatch.setenv(_ENV, "gurobi")
    assert get_root_provider() == "builtin"


# --- set_root_provider ----------------------------------------------------------------------------


def test_set_builtin_round_trips() -> None:
    """Setting ``builtin`` is what ``get_root_provider`` then reports."""
    set_root_provider("builtin")
    assert get_root_provider() == "builtin"


def test_set_unknown_name_raises_and_changes_nothing() -> None:
    """An unknown name is a ``ValueError`` that leaves the active provider alone."""
    set_root_provider("builtin")
    with pytest.raises(ValueError, match="Unknown root provider"):
        set_root_provider("gurobi")
    with pytest.raises(ValueError, match="Unknown root provider"):
        set_root_provider("")
    assert get_root_provider() == "builtin"


@pytest.mark.parametrize("bad", [1, None, b"builtin", ["builtin"]], ids=repr)
def test_set_non_string_raises_type_error(bad: object) -> None:
    """A non-string name is a ``TypeError`` (not a ``ValueError`` about an unknown name)."""
    set_root_provider("builtin")
    with pytest.raises(TypeError, match="must be a str"):
        set_root_provider(bad)  # type: ignore[arg-type]
    assert get_root_provider() == "builtin"


def test_set_berf_without_the_package_raises_import_error() -> None:
    """``berf`` needs ``pantr-berf``: an ``ImportError`` that names it, with the state unchanged."""
    set_root_provider("builtin")
    with pytest.raises(ImportError, match="pantr-berf"):
        set_root_provider("berf")
    assert get_root_provider() == "builtin"


# --- Routing, both ways ---------------------------------------------------------------------------


def test_switching_routes_every_entry_point_to_the_provider_and_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each public function goes to the active provider's module, in the dtype's twin.

    Bug caught: a function that caches the provider module at import, ignores the switch, or picks
    the ``float64`` function for ``float32`` coefficients. The fake answers differently from the
    Builtin, so the route is visible in the result.
    """
    fake = _FakeBerf()
    _install_fake(monkeypatch, fake)
    line64 = _line()
    line32 = Bezier(np.array([1.0, -1.0], dtype=np.float32))
    set_root_provider("builtin")
    np.testing.assert_allclose(find_roots(line64), [0.5], atol=1e-12)
    assert fake.calls == []

    set_root_provider("berf")
    assert get_root_provider() == "berf"
    np.testing.assert_array_equal(find_roots(line64), [0.25, 0.75])
    np.testing.assert_array_equal(find_roots(line32), [0.25, 0.75])
    roots, counts = find_roots([line64, line64])
    np.testing.assert_array_equal(counts, [1, 1])
    np.testing.assert_array_equal(roots[:, 0], [0.25, 0.25])
    find_roots([line32, line32])
    assert find_monotone_root(line64) == 0.5
    assert find_monotone_root(line32) == 0.5
    np.testing.assert_array_equal(find_monotone_root([line64, line64]), [0.5, 0.5])
    np.testing.assert_array_equal(find_monotone_root([line32, line32]), [0.5, 0.5])
    assert [c[0] for c in fake.calls] == [
        "find_roots",
        "find_roots_f32",
        "find_roots_batch",
        "find_roots_batch_f32",
        "find_monotone_root",
        "find_monotone_root_f32",
        "find_monotone_root_batch",
        "find_monotone_root_batch_f32",
    ]

    fake.calls.clear()
    set_root_provider("builtin")
    assert get_root_provider() == "builtin"
    np.testing.assert_allclose(find_roots(line64), [0.5], atol=1e-12)
    np.testing.assert_allclose(find_roots(line32), [0.5], atol=1e-6)
    assert find_monotone_root(line64) == pytest.approx(0.5, abs=1e-12)
    assert fake.calls == []


def test_provider_receives_contiguous_coefficients_and_the_resolved_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provider gets C-contiguous arrays of the dtype and one resolved tolerance for both uses.

    The arrays are ``(degree + 1,)`` for one polynomial and ``(n, degree + 1)`` for a batch;
    ``param_tol`` and ``value_tol`` are the same number.

    Bug caught: a strided view of the control points passed through (a rank-1 slice of a
    ``(degree + 1, 1)`` array is contiguous only by accident of the column count), the user's
    ``tol`` replaced by the default, or the two tolerances differing.
    """
    fake = _FakeBerf()
    _install_fake(monkeypatch, fake)
    set_root_provider("berf")
    coeff = np.array([3.0, -1.0, 0.5, 2.0])
    find_roots(Bezier(coeff), tol=1e-9)
    find_roots(Bezier(coeff))
    find_roots([Bezier(coeff), Bezier(coeff[::-1].copy())], tol=2e-9)
    find_monotone_root(Bezier(coeff), tol=3e-9)
    rational = Bezier(np.array([[3.0, 1.0], [-1.0, 2.0], [0.5, 1.5]]), is_rational=True)
    find_roots(rational, tol=4e-9)
    by_name = {c[0]: c for c in fake.calls}
    assert all(c[3] for c in fake.calls)  # C-contiguous
    assert all(c[1] == np.float64 for c in fake.calls)
    assert fake.calls[0][2:5] == ((4,), True, (1e-9, 1e-9))
    default = float(get_strict(np.float64))
    assert fake.calls[1][4] == (default, default)
    assert by_name["find_roots_batch"][2] == (2, 4)
    assert by_name["find_roots_batch"][4] == (2e-9, 2e-9)
    assert by_name["find_monotone_root"][4] == (3e-9,)
    # The rational curve's first homogeneous column reaches the provider, as a (3,) array.
    assert fake.calls[-1][2] == (3,)
    assert fake.calls[-1][4] == (4e-9, 4e-9)


# --- The incompleteness warning (QB7) ------------------------------------------------------------


def test_depth_cap_input_warns_and_still_returns_the_root() -> None:
    """The real depth-cap path: the root is found, and the result says it is not complete.

    Bug caught: the previous release dropped such work without a word. A tolerance of 1e-52 is below
    the spacing of doubles at the root, so no interval narrows to it; the wrapper must raise a
    ``RuntimeWarning`` naming the subdivision budget, attributed to the caller (``stacklevel``),
    and the root must still be reported.
    """
    with pytest.warns(RuntimeWarning, match="subdivision budget") as record:
        roots = find_roots(Bezier(_CAP_COEFFS), tol=_CAP_TOL)
    assert len(record) == 1
    assert "1 of 1 polynomials" in str(record[0].message)
    assert Path(record[0].filename).name == Path(__file__).name
    assert roots.shape == (1,)
    assert abs(roots[0] - _CAP_ROOT) <= 4.0 * _EPS


def test_depth_cap_input_reports_its_gap_through_the_detailed_interface() -> None:
    """The same input, below the wrapper: ``gap_localized`` with one gap around the root."""
    roots, claims, _multiplicities, completeness, gaps = find_roots_detailed(
        _CAP_COEFFS, _CAP_TOL, _CAP_TOL
    )
    assert completeness == ext.roots.GAP_LOCALIZED
    assert gaps.shape == (1, 2)
    assert gaps[0, 0] <= _CAP_ROOT + 4.0 * _EPS
    assert gaps[0, 1] >= _CAP_ROOT - 4.0 * _EPS
    assert roots.shape == (1,)
    assert claims.tolist() == [ext.roots.CLAIM_COMPUTED]


def test_batch_with_one_capped_polynomial_warns_once_for_that_one() -> None:
    """A batch warns once, counting the polynomials with a gap (here 1 of 3).

    The other rows have no root in [0, 1] (all coefficients positive), which the tolerance cannot
    push to the cap: clipping rejects them at the first test.
    """
    positive = [Bezier(np.linspace(1.0, 2.0 + k, 9)) for k in range(2)]
    with pytest.warns(RuntimeWarning, match="subdivision budget") as record:
        roots, counts = find_roots([Bezier(_CAP_COEFFS), *positive], tol=_CAP_TOL)
    assert len(record) == 1
    assert "1 of 3 polynomials" in str(record[0].message)
    assert Path(record[0].filename).name == Path(__file__).name
    np.testing.assert_array_equal(counts, [1, 0, 0])
    assert abs(roots[0, 0] - _CAP_ROOT) <= 4.0 * _EPS
    assert np.isnan(roots[1:]).all()


@pytest.mark.parametrize("code", [1, 2], ids=["gap_localized", "gap_unlocalized"])
def test_a_provider_result_with_any_gap_warns(monkeypatch: pytest.MonkeyPatch, code: int) -> None:
    """Any completeness other than ``complete`` becomes the warning, whoever returned it.

    Bug caught: the wrapper testing for ``gap_localized`` only, so a provider that cannot say where
    its gap is (``gap_unlocalized``) passes as complete.
    """
    _install_fake(monkeypatch, _FakeBerf(completeness=code))
    set_root_provider("berf")
    with pytest.warns(RuntimeWarning, match="1 of 1 polynomials"):
        find_roots(_line())
    _install_fake(monkeypatch, _FakeBerf(completeness=[0, code]))
    with pytest.warns(RuntimeWarning, match="1 of 2 polynomials"):
        find_roots([_line(), _line()])


def test_a_batch_warning_counts_only_the_incomplete_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rows 1 and 2 of 4 incomplete: one warning, "2 of 4"."""
    _install_fake(monkeypatch, _FakeBerf(completeness=[0, 1, 2, 0]))
    set_root_provider("berf")
    with pytest.warns(RuntimeWarning, match="2 of 4 polynomials") as record:
        find_roots([_line() for _ in range(4)])
    assert len(record) == 1


def test_a_complete_result_does_not_warn(monkeypatch: pytest.MonkeyPatch) -> None:
    """No warning when every result is complete, from either provider and in every mode."""
    _install_fake(monkeypatch, _FakeBerf(completeness=0))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for provider in ("builtin", "berf"):
            set_root_provider(provider)
            find_roots(_line())
            find_roots([_line(), _line()])
            find_monotone_root(_line())
            find_monotone_root([_line(), _line()])


def test_rational_scalar_curve_is_solved_on_its_first_homogeneous_column() -> None:
    """The numerator ``x * w`` is solved: weights that differ from it do not move the roots.

    Bug caught: the weight column, or the quotient, handed to the provider. The numerator
    (-1/2 + t) has the root 1/2 whatever the (positive) weights.
    """
    numerator = np.array([-0.5, 0.5])
    weights = np.array([1.0, 3.0])
    rational = Bezier(np.stack([numerator, weights], axis=1), is_rational=True)
    np.testing.assert_allclose(find_roots(rational), [0.5], atol=1e-12)
    np.testing.assert_allclose(find_roots(rational), find_roots(Bezier(numerator)), atol=1e-12)
