"""Determinism of the THB port across thread counts and instruction-set rungs.

The THB code runs its per-cell tables, evaluations, prolongation columns, quasi-interpolation
contractions, extraction blocks and graph rows in parallel loops whose chunking is a function of
the problem only. What a thread count or an instruction-set rung may change is the last bits of a
floating-point result, never a discrete one:

* every discrete output (active sets of every level and cell, dof owners, the halo, the coupling
  graph, the prolongation's sparsity pattern, the extraction's active sets) is identical at one
  thread and at several, and in a fresh process on the generic rung;
* every floating-point output (basis, spline and operator values, quasi-interpolation
  coefficients, prolongation entries) agrees within a derived bound.

The oracle is the same code run under another setting: there is no closed form for these outputs,
and the properties they must satisfy are checked in ``test_thb_port.py``. Both snapshots are taken
on fresh spaces, so a lazily built table is built under the setting being tested.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import numba as nb
import numpy as np
import pytest

import pantr
from pantr._ext import ext
from pantr.tolerance import get_conservative
from tests import _thb_snapshot as snap
from tests._tree_env import subprocess_env

_REPO_ROOT = Path(__file__).resolve().parents[1]

_VALUE_RTOL = float(get_conservative(np.float64))
"""Relative bound on a floating-point output moved by a setting that changes no algorithm.

The outputs are contractions of up to a few tens of terms with weights of order one to a few
tens (the quasi-interpolation functionals of a degree-3 direction are the largest), plus an SVD
solve per prolongation column whose backward error the library itself grades at this tier
(``pantr::thb::prolongation_residual_tolerance``): a long accumulation, the ``get_conservative``
tier (4096 eps). Scaled by ``max(1, |largest value|)`` of the output compared. A fused
multiply-add (the generic rung has none) or another summation order moves results by a small
multiple of eps, far inside it.
"""


def _many_threads() -> int:
    """Pick a thread count above one for the parallel run.

    Returns:
        int: Eight, capped at the most threads ``pantr.set_num_threads`` accepts and at the C++
        pool's capacity.
    """
    numba_cap: int = nb.config.NUMBA_NUM_THREADS
    return min(8, numba_cap, ext.max_threads())


def _assert_same(label: str, left: NamedTuple, right: NamedTuple) -> None:
    """Compare two snapshots field by field: integers exactly, floats within the bound.

    Args:
        label: Names the pair in a failure message.
        left: The first snapshot.
        right: The second, of the same type.
    """
    assert left._fields == right._fields
    for name, a, b in zip(left._fields, left, right, strict=True):
        where = f"{label}.{name}"
        assert a.shape == b.shape, f"{where}: shapes {a.shape} and {b.shape}"
        assert a.size > 0, f"{where} is empty: the comparison would say nothing"
        if np.issubdtype(a.dtype, np.integer):
            np.testing.assert_array_equal(a, b, err_msg=where)
        else:
            bound = _VALUE_RTOL * max(1.0, float(np.abs(a).max()))
            gap = float(np.abs(a - b).max())
            assert gap <= bound, f"{where}: max difference {gap:.3e} > {bound:.3e}"


def _take_both() -> tuple[snap.Snapshot, snap.ProlongationSnapshot]:
    """Take the large space's snapshot and the small pair's prolongation, on fresh spaces."""
    return snap.take_snapshot(snap.large_space()), snap.take_prolongation(*snap.refinement_pair())


class TestThreadCount:
    """One thread and several give the same discrete outputs and the same values."""

    def test_outputs_do_not_depend_on_the_thread_count(self) -> None:
        many = _many_threads()
        if many < 2:
            pytest.skip("needs at least two threads")
        pantr.set_num_threads(1)
        serial_space, serial_prolongation = _take_both()
        pantr.set_num_threads(many)
        parallel_space, parallel_prolongation = _take_both()

        _assert_same("space", serial_space, parallel_space)
        _assert_same("prolongation", serial_prolongation, parallel_prolongation)
        # The fixtures exercise what the comparison is about: dead dofs, a halo, a graph.
        assert (parallel_space.dof_owner < 0).any()
        assert parallel_space.halo.size > 0
        assert parallel_space.level_sizes.size == 3

    def test_the_order_of_the_settings_does_not_matter(self) -> None:
        """Several threads first, then one: the first use of a lazy table is not the difference."""
        many = _many_threads()
        if many < 2:
            pytest.skip("needs at least two threads")
        pantr.set_num_threads(many)
        parallel_space, parallel_prolongation = _take_both()
        pantr.set_num_threads(1)
        serial_space, serial_prolongation = _take_both()

        _assert_same("space", parallel_space, serial_space)
        _assert_same("prolongation", parallel_prolongation, serial_prolongation)


class TestGenericRung:
    """A fresh process on the generic rung gives the same discrete outputs and the same values."""

    def test_the_generic_rung_agrees_with_this_process(self, tmp_path: Path) -> None:
        env = subprocess_env()
        env["PANTR_ISA"] = "generic"
        out = tmp_path / "generic.npz"
        result = subprocess.run(
            [sys.executable, "-m", "tests._thb_snapshot", str(out)],
            capture_output=True,
            text=True,
            check=False,
            env=env,
            cwd=_REPO_ROOT,
        )
        assert result.returncode == 0, result.stderr
        # The child names its rung and the file it imported, so the comparison is against the
        # pantr this process imported (this checkout's or the installed wheel's) on the generic
        # rung.
        rung, imported = result.stdout.split("\n")[:2]
        assert rung == "generic"
        assert Path(imported).resolve() == Path(pantr.__file__).resolve(), imported

        space_here, prolongation_here = snap.take_small()
        space_there, prolongation_there = snap.load(out)
        _assert_same("space", space_here, space_there)
        _assert_same("prolongation", prolongation_here, prolongation_there)
