"""Constructive operations: extrusion, revolution, ruled, and sweep.

Provides functions that create higher-dimensional B-spline objects
by combining existing ones: extrusion along a vector, revolution
around an axis, ruled interpolation, and translational sweep.
"""

from __future__ import annotations

import numpy as np
from numpy import typing as npt

from .._ext import ext
from ..bspline import Bspline, BsplineSpace
from ._primitives import _linear_space_1d, _points, _sweep_pair
from ._validation import _PHYSICAL_DIM, _pad_to_3d, _promote_to_rational

_MAX_DIM_FOR_OPERATIONS = 2


def create_extrusion(bspline: Bspline, displacement: npt.ArrayLike) -> Bspline:
    """Extrude a B-spline curve or surface along a displacement vector.

    Creates a new B-spline with one additional parametric dimension by
    translating the input along the given vector.  The new direction is
    appended as the last parametric axis with degree 1 and knots
    ``[0, 0, 1, 1]``.

    Args:
        bspline: Input curve (dim=1) or surface (dim=2).
        displacement: Translation vector (up to 3D, zero-padded).

    Returns:
        Bspline: A B-spline with ``dim + 1`` parametric dimensions.

    Raises:
        ValueError: If ``bspline.dim > 2``.

    Example:
        >>> from pantr.cad import create_circle, create_extrusion
        >>> cyl = create_extrusion(create_circle(), [0, 0, 1])
        >>> cyl.dim
        2
    """
    if bspline.dim > _MAX_DIM_FOR_OPERATIONS:
        raise ValueError(
            f"create_extrusion requires dim <= {_MAX_DIM_FOR_OPERATIONS}, got {bspline.dim}."
        )

    disp = _pad_to_3d(displacement)
    cp = bspline.control_points
    orig_shape = cp.shape[:-1]  # (*num_basis,)
    rank_full = cp.shape[-1]

    new_cp = np.empty((*orig_shape, 2, rank_full), dtype=cp.dtype)
    new_cp[..., 0, :] = cp

    if bspline.is_rational:
        # Weighted homogeneous: (w*x, w*y, w*z, w)
        # Translated: (w*x + w*dx, w*y + w*dy, w*z + w*dz, w)
        new_cp[..., 1, :] = cp
        weights = cp[..., _PHYSICAL_DIM : _PHYSICAL_DIM + 1]
        new_cp[..., 1, :_PHYSICAL_DIM] = cp[..., :_PHYSICAL_DIM] + weights * disp
    else:
        new_cp[..., 1, :] = cp + disp[:rank_full].astype(cp.dtype)

    spaces = [*bspline.space.spaces, _linear_space_1d(dtype=bspline.dtype)]
    new_space = BsplineSpace(spaces)
    return Bspline(new_space, new_cp, is_rational=bspline.is_rational)


def create_ruled(bspline1: Bspline, bspline2: Bspline) -> Bspline:
    """Construct a ruled surface or volume between two B-splines.

    Creates a new B-spline by linearly interpolating control points
    between *bspline1* (at parameter 0) and *bspline2* (at parameter 1)
    along a new last parametric axis with degree 1.

    The two inputs are first made compatible via :func:`make_compat` so they
    share the same degree and knot vectors.  If one is rational and the
    other is not, the non-rational one is promoted.

    Args:
        bspline1: First boundary (curve or surface, dim <= 2).
        bspline2: Second boundary (same dim as *bspline1*).

    Returns:
        Bspline: A B-spline with ``dim + 1`` parametric dimensions.

    Raises:
        ValueError: If the inputs have different parametric dimensions.
        ValueError: If either input has ``dim > 2``.

    Example:
        >>> from pantr.cad import create_circle, create_ruled
        >>> annulus = create_ruled(create_circle(radius=0.5), create_circle(radius=1.0))
        >>> annulus.dim
        2
    """
    return Bspline._from_handle(ext.cad.create_ruled(bspline1._h, bspline2._h))


def _axis_vector(axis: int | npt.ArrayLike) -> list[float]:
    """Read an axis specification as the direction the compiled function takes.

    Args:
        axis (int | npt.ArrayLike): An ``int`` (coordinate axis) or array-like direction.

    Returns:
        list[float]: The unit vector of a coordinate axis, or the direction's coordinates (the
        compiled function zero-pads, checks and normalizes them).

    Raises:
        IndexError: If an ``int`` is outside ``[-3, 3)``.
    """
    if isinstance(axis, int | np.integer):
        v = np.zeros(_PHYSICAL_DIM, dtype=np.float64)
        v[int(axis)] = 1.0
        return v.tolist()
    return np.asarray(axis, dtype=np.float64).ravel().tolist()


def create_revolution(
    bspline: Bspline,
    point: npt.ArrayLike,
    axis: int | npt.ArrayLike = 2,
    angle: float | tuple[float, float] | None = None,
) -> Bspline:
    """Revolve a B-spline curve or surface around an axis.

    Creates a new B-spline with one additional parametric dimension
    (the angular direction, appended last).  The input is first
    promoted to rational if needed, then transformed to a coordinate
    system aligned with the rotation axis, revolved via the circular
    arc construction, and transformed back.

    The angular direction inherits the same span/continuity structure
    as :func:`create_circle`: one span per 90 degrees, C0 at arc junctions.

    Args:
        bspline: Input curve (dim=1) or surface (dim=2) with rank 3.
        point: A point on the rotation axis (up to 3D, zero-padded).
        axis: Rotation axis.  An ``int`` in ``{0, 1, 2}`` selects
            a coordinate axis.  An array-like of length 3 specifies
            an arbitrary axis direction (normalised internally).
        angle: Sweep specification (same as :func:`create_circle`).

    Returns:
        Bspline: A rational B-spline with ``dim + 1`` dimensions.

    Raises:
        ValueError: If ``bspline.dim > 2``.
        ValueError: If ``bspline.rank != 3``.
    """
    h = ext.cad.create_revolution(
        bspline._h, _points(point), _axis_vector(axis), _sweep_pair(angle)
    )
    return Bspline._from_handle(h)


def create_sweep(section: Bspline, trajectory: Bspline) -> Bspline:
    """Construct the translational sweep of a section along a trajectory.

    Creates a new B-spline by summing the section and trajectory
    geometries: ``S(u, v) = section(u) + trajectory(v)``.  The
    trajectory direction is appended as the last parametric axis.

    For rational B-splines the product formula applies: the result
    weight is the product of section and trajectory weights, and the
    weighted coordinates combine as
    ``w_s * C_t + w_t * C_s``.

    Args:
        section: Section curve (dim=1) or surface (dim=2).
        trajectory: Trajectory curve (dim=1).

    Returns:
        Bspline: A B-spline with ``section.dim + 1`` dimensions.

    Raises:
        ValueError: If ``section.dim > 2``.
        ValueError: If ``trajectory.dim != 1``.
    """
    if section.dim > _MAX_DIM_FOR_OPERATIONS:
        raise ValueError(
            f"create_sweep requires section.dim <= {_MAX_DIM_FOR_OPERATIONS}, got {section.dim}."
        )
    if trajectory.dim != 1:
        raise ValueError(f"create_sweep requires trajectory.dim == 1, got {trajectory.dim}.")

    is_rational = section.is_rational or trajectory.is_rational

    if is_rational:
        sec = _promote_to_rational(section)
        traj = _promote_to_rational(trajectory)
        cp_s = sec.control_points
        cp_t = traj.control_points

        ws = cp_s[..., _PHYSICAL_DIM]
        wt = cp_t[..., _PHYSICAL_DIM]
        cs = cp_s[..., :_PHYSICAL_DIM]
        ct = cp_t[..., :_PHYSICAL_DIM]

        # Weighted coords: w_t * C_s + w_s * C_t
        term_s = cs[..., np.newaxis, :] * wt[..., np.newaxis]
        term_t = ct * ws[..., np.newaxis, np.newaxis]
        new_coords = term_s + term_t
        new_weights = ws[..., np.newaxis] * wt

        new_cp = np.concatenate(
            [new_coords, new_weights[..., np.newaxis]],
            axis=-1,
        )
        spaces = [*sec.space.spaces, *traj.space.spaces]
    else:
        cp_s = section.control_points
        cp_t = trajectory.control_points
        new_cp = cp_s[..., np.newaxis, :] + cp_t
        spaces = [*section.space.spaces, *trajectory.space.spaces]

    new_space = BsplineSpace(spaces)
    return Bspline(new_space, new_cp, is_rational=is_rational)
