"""A negative ``tol`` is refused, ``tol=0`` fits finitely, in every fit and in interface detection.

Before, a finite negative ``tol`` was accepted: the truncated-SVD solves then kept an exactly zero
singular value and divided by it (all-NaN control points), and ``detect_interfaces`` matched nothing
at a negative relative tolerance. ``fit_bezier(tol=0)`` kept an exactly zero singular value as well
(its truncation compares ``s >= tol * s_max``), while ``fit_bspline(tol=0)`` was finite.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from pantr._ext import ext
from pantr.bezier import fit_bezier, interpolate_bezier
from pantr.bspline import (
    Bspline,
    BsplineSpace,
    BsplineSpace1D,
    create_greville_lattice,
    create_uniform_space,
    fit_bspline,
    interpolate_bspline,
    l2_project_bspline,
)
from pantr.grid import partition_grid, tensor_product_grid
from pantr.mpi import (
    DistributedSpace,
    fit_bspline_distributed,
    interpolate_bspline_distributed,
    l2_project_bspline_distributed,
)
from pantr.multipatch import MultiPatch, detect_interfaces

NEGATIVE_TOLS = [-1.0, -1e-9]


def _message(tol: float) -> str:
    """The refusal's pattern for ``tol``: the argument, the requirement and the value."""
    return rf"^tol must be finite and non-negative, got {re.escape(str(tol))}"


def _bspline_system() -> tuple[BsplineSpace, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Twelve nodes in ``[0, 0.4]`` for a 7-function space: the right-hand functions vanish.

    Returns:
        tuple: The space, the ``(12, 1)`` nodes and the values; the collocation matrix has
        exactly zero columns, so its truncated SVD has exactly zero singular values.
    """
    space = create_uniform_space(3, 4)
    nodes = np.linspace(0.0, 0.4, 12).reshape(-1, 1)
    return space, nodes, np.sin(nodes[:, 0])


def _two_patches() -> list[Bspline]:
    """Two bilinear patches sharing one edge exactly."""
    sq = BsplineSpace([BsplineSpace1D([0, 0, 1, 1], 1)] * 2)
    left = Bspline(sq, np.array([[0.0, 0], [0, 1], [1, 0], [1, 1]]))
    right = Bspline(sq, np.array([[1.0, 0], [1, 1], [2, 0], [2, 1]]))
    return [left, right]


# ------------------------------------------------------------------------------ the reproduction


@pytest.mark.parametrize("layout", ["scattered", "tensor"])
@pytest.mark.parametrize("tol", [None, 0.0])
def test_fit_bspline_is_finite_for_no_tol_and_zero_tol(tol: float | None, layout: str) -> None:
    """Controls: the default and ``tol=0`` give finite control points on the rank-deficient fit."""
    space, nodes, values = _bspline_system()
    given: Any = nodes if layout == "scattered" else [nodes[:, 0]]
    cp = np.asarray(fit_bspline(values, given, space, tol=tol).control_points)
    assert np.isfinite(cp).all()


@pytest.mark.parametrize("layout", ["scattered", "tensor"])
@pytest.mark.parametrize("tol", NEGATIVE_TOLS)
def test_fit_bspline_refuses_a_negative_tol(tol: float, layout: str) -> None:
    """Bug caught: a negative tol kept the exactly zero singular values, all-NaN control points."""
    space, nodes, values = _bspline_system()
    given: Any = nodes if layout == "scattered" else [nodes[:, 0]]
    with pytest.raises(ValueError, match=_message(tol)):
        fit_bspline(values, given, space, tol=tol)


@pytest.mark.parametrize("layout", ["scattered", "tensor"])
@pytest.mark.parametrize("tol", [None, 0.0])
def test_fit_bezier_is_finite_for_no_tol_and_zero_tol(tol: float | None, layout: str) -> None:
    """Bug caught: ``tol=0`` kept an exactly zero singular value of the rank-1 system.

    Six samples at one node and degree 3 give a rank-1 Bernstein matrix; the zero singular values
    were inverted. The oracle is exact: the minimum-norm least-squares fit of the constant 1
    through the node 0 has the first Bernstein coefficient 1 and the others 0.
    """
    nodes: Any = np.zeros((6, 1)) if layout == "scattered" else [np.zeros(6)]
    cp = np.asarray(fit_bezier(np.ones(6), nodes, degree=3, tol=tol).control_points).ravel()
    assert np.isfinite(cp).all()
    np.testing.assert_allclose(cp, [1.0, 0.0, 0.0, 0.0], rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("layout", ["scattered", "tensor"])
@pytest.mark.parametrize("tol", NEGATIVE_TOLS)
def test_fit_bezier_refuses_a_negative_tol(tol: float, layout: str) -> None:
    """Bug caught: a negative tol returned all-NaN control points for the rank-1 system."""
    nodes: Any = np.zeros((6, 1)) if layout == "scattered" else [np.zeros(6)]
    with pytest.raises(ValueError, match=_message(tol)):
        fit_bezier(np.ones(6), nodes, degree=3, tol=tol)


def test_fit_bezier_tol_zero_is_finite_in_two_dimensions_with_duplicated_nodes() -> None:
    """The tensor form: duplicated nodes in each of two directions, ``tol=0`` stays finite."""
    nodes = [np.zeros(5), np.array([0.0, 0.0, 0.5, 0.5, 1.0, 1.0])]
    values = np.ones((5, 6))
    cp = np.asarray(fit_bezier(values, nodes, degree=(2, 3), tol=0.0).control_points)
    assert np.isfinite(cp).all()


@pytest.mark.parametrize("tol", [None, 0.0])
def test_detect_interfaces_finds_the_shared_edge_for_no_tol_and_zero_tol(
    tol: float | None,
) -> None:
    """Controls: the default and ``tol=0`` find the one interface of an exactly shared edge."""
    assert len(detect_interfaces(_two_patches(), tol=tol)) == 1


@pytest.mark.parametrize("tol", NEGATIVE_TOLS)
def test_detect_interfaces_refuses_a_negative_tol(tol: float) -> None:
    """Bug caught: a negative relative tolerance matched nothing, an empty list as the answer."""
    with pytest.raises(ValueError, match=_message(tol)):
        detect_interfaces(_two_patches(), tol=tol)
    with pytest.raises(ValueError, match=_message(tol)):
        detect_interfaces([], tol=tol)  # checked before the empty-input shortcut


@pytest.mark.parametrize("tol", NEGATIVE_TOLS)
def test_multipatch_refuses_a_negative_tol(tol: float) -> None:
    """MultiPatch forwards ``tol`` to detection and verification, which refuse it."""
    patches = _two_patches()
    interfaces = detect_interfaces(patches)
    with pytest.raises(ValueError, match=_message(tol)):
        MultiPatch(patches, tol=tol)
    with pytest.raises(ValueError, match=_message(tol)):
        MultiPatch(patches, interfaces, tol=tol)


# ------------------------------------------------------------ the other entry points taking tol


@pytest.mark.parametrize("tol", NEGATIVE_TOLS)
def test_callable_fits_refuse_a_negative_tol(tol: float) -> None:
    """The callable forms share the check, and refuse before sampling (see the sampling test)."""
    space = create_uniform_space(3, 4)
    with pytest.raises(ValueError, match=_message(tol)):
        interpolate_bspline(lambda lat: lat.pts_per_dir[0] ** 2, space, tol=tol)
    with pytest.raises(ValueError, match=_message(tol)):
        l2_project_bspline(lambda lat: lat.pts_per_dir[0] ** 2, space, tol=tol)
    with pytest.raises(ValueError, match=_message(tol)):
        interpolate_bezier(lambda lat: lat.pts_per_dir[0] ** 2, 8, degree=3, tol=tol)


@pytest.mark.parametrize("tol", NEGATIVE_TOLS)
def test_compiled_fits_refuse_a_negative_tol(tol: float) -> None:
    """The compiled Bezier fits refuse it themselves, as they do a NaN."""
    x = np.linspace(0.0, 1.0, 4)
    with pytest.raises(ValueError, match=_message(tol)):
        ext.bezier.fit_tensor(x[:, np.newaxis] ** 2, [x], (3,), tol)
    with pytest.raises(ValueError, match=_message(tol)):
        ext.bezier.fit_scattered(x[:, np.newaxis] ** 2, x[:, np.newaxis], [3], tol)


class _OneRank:
    """A one-rank stand-in for an mpi4py communicator."""

    rank = 0
    size = 1

    def allreduce(self, data: Any) -> Any:
        """Return the one rank's contribution, which is the sum.

        Args:
            data (Any): The rank's contribution.

        Returns:
            Any: ``data``.
        """
        return data

    def allgather(self, data: Any) -> list[Any]:
        """Return the one rank's contribution in a list.

        Args:
            data (Any): The rank's contribution.

        Returns:
            list[Any]: ``[data]``.
        """
        return [data]


@pytest.mark.parametrize("tol", NEGATIVE_TOLS)
def test_distributed_wrappers_refuse_a_negative_tol(tol: float) -> None:
    """The MPI wrappers forward ``tol`` to the same solves and refuse a negative one."""
    space = create_uniform_space(3, 4)
    ds = DistributedSpace(space, partition_grid(tensor_product_grid(space), 1), _OneRank())
    lattice = create_greville_lattice(space)
    values = np.ones(7)
    with pytest.raises(ValueError, match=_message(tol)):
        interpolate_bspline_distributed(lambda pts: pts[:, 0] ** 2, ds, tol=tol)
    with pytest.raises(ValueError, match=_message(tol)):
        l2_project_bspline_distributed(lambda lat: lat.pts_per_dir[0] ** 2, ds, tol=tol)
    with pytest.raises(ValueError, match=_message(tol)):
        fit_bspline_distributed(values, lattice, ds, tol=tol)
    with pytest.raises(ValueError, match=_message(tol)):
        fit_bspline_distributed(values, lattice, ds, values_distributed=True, tol=tol)
