"""Root finding for scalar B-splines by the Mørken-Reimers method.

Holds the public :func:`find_roots` (Layer 1). The validation, the tolerances, the
iteration and the merge of repeated reports are in the compiled library
(``include/pantr/bspline/roots.hpp``), which returns how the iteration ended as a status
and two counts; this module turns them into the two :class:`RuntimeWarning` messages.

The method computes the zeros of a scalar spline on the spline's own knot vector
by repeatedly inserting the zero of the control polygon as a new knot
:cite:p:`morken2007zeros`. Compared with extracting the Bézier segments and
solving each of them (:func:`pantr.bezier.find_roots`), it needs no extraction
and no stitching of roots at segment boundaries, zeros sitting exactly on a knot
need no special case, and convergence is unconditional rather than heuristically
controlled.

**Precision contract.** Knots and coefficients are promoted to float64 before
the iteration, which is exact for a float32 spline, so the arithmetic is float64
throughout regardless of the input dtype. The tolerances, however, are resolved
from the *input* dtype: a float32 spline carries float32-level information, and
a residual below that level cannot be distinguished from zero no matter how the
iteration is carried out.

**What is found.** The control polygon says where to look: a spline whose
coefficients have no sign change has no zeros at all, so every zero where the
spline changes sign is bracketed by a sign change and is searched for. What it
does not do is certify. The tracking of a sign change stops for three different
reasons, and none of them implies that the point it stopped at is a zero, so a
value reaches the result only when its residual ``|f(x)|`` is within what a zero
may legitimately leave behind: the error of evaluating the spline, plus the value
the spline reaches at the parametric resolution the iteration stops at,
``|f'| * 2 * tol * scale``. That residual test is the only way a zero of even
multiplicity can be reported at all, since the sign changes bracketing it
disappear under refinement. The two domain endpoints are decided on the
evaluation error alone, being read straight off a coefficient with no location
error to allow for.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy import typing as npt

if TYPE_CHECKING:
    from pantr.bspline._bspline import Bspline


def _find_roots_impl(
    bspline: Bspline,
    *,
    tol: float | None = None,
    max_insertions: int | None = None,
) -> npt.NDArray[np.float64]:
    """Layer 2 implementation for :func:`find_roots`.

    Args:
        bspline (Bspline): Scalar univariate B-spline.
        tol (float | None): Relative parametric tolerance, or ``None`` for the
            dtype default.
        max_insertions (int | None): Insertion budget per zero, or ``None`` for the
            degree's default. Only tests pass one, to reach the exhaustion paths.

    Returns:
        npt.NDArray[np.float64]: Sorted, read-only array of roots.

    Raises:
        TypeError: If ``bspline`` is not a :class:`~pantr.bspline.Bspline`.
        ValueError: If ``tol`` is not positive, or if the spline is not scalar and
            univariate, has degree zero, has a non-positive weight, is identically
            zero or vanishes identically on a knot interval.
    """
    from pantr.bspline._bspline import Bspline as BsplineCls  # noqa: PLC0415

    # Annotated as a Bspline, but called with anything: the check is for that caller.
    candidate: object = bspline
    if not isinstance(candidate, BsplineCls):
        msg = f"Expected a Bspline instance, got {type(candidate).__name__}"
        raise TypeError(msg)
    if tol is not None and tol <= 0.0:
        msg = f"tol must be positive, got {tol}"
        raise ValueError(msg)

    handle: Any = bspline._h
    roots, _status, count, truncated, abandoned, budget, root_tol = handle.find_roots(
        None if tol is None else float(tol), max_insertions
    )
    if truncated > 0:
        warnings.warn(
            f"{truncated} of {count} roots exhausted their insertion budget of "
            f"{budget} before the iterates stagnated; they are reported at the last "
            "iterate reached, so they may be less accurate than tol asks for",
            RuntimeWarning,
            stacklevel=3,
        )
    if abandoned > 0:
        warnings.warn(
            f"{abandoned} sign changes exhausted their insertion budget of {budget} "
            f"without the residual ever falling to {root_tol:.3e}, so they are not "
            "reported; the result may be missing a root",
            RuntimeWarning,
            stacklevel=3,
        )
    result: npt.NDArray[np.float64] = roots
    result.flags.writeable = False
    return result


def find_roots(
    bspline: Bspline,
    *,
    tol: float | None = None,
) -> npt.NDArray[np.float64]:
    """Find every zero of a scalar univariate B-spline.

    Uses the Mørken-Reimers method: the zero of the control polygon is inserted
    as a new knot, over and over, until the iterates stagnate. The iteration
    needs no starting value, converges for any spline, and reaches a simple zero
    quadratically -- once per ``degree - 1`` insertions, not once per insertion.

    Args:
        bspline (Bspline): A univariate (``dim == 1``), scalar-valued
            (``rank == 1``) B-spline. Periodic or unclamped splines are
            converted to an open representation first. For rational splines the
            zeros are those of the numerator, which are the zeros of the
            mapping because the weights are positive.
        tol (float | None): Relative parametric tolerance. The iteration stops
            when the spread of the last ``degree`` iterates falls to
            ``tol * max(|t_a|, |t_{a+degree}|, domain_length)``, so that it
            transforms correctly when the parametric domain is scaled or moved
            away from the origin. Defaults to
            ``tolerance.get_strict(bspline.dtype)``.

    Returns:
        npt.NDArray[np.float64]: Sorted, read-only array of roots, always
        float64. Empty if the spline has no zero.

    Raises:
        TypeError: If ``bspline`` is not a :class:`~pantr.bspline.Bspline`.
        ValueError: If ``bspline`` is not univariate and scalar valued, has
            degree zero, is rational with a non-positive weight, vanishes
            identically on a knot interval, or if ``tol`` is not positive.

    Warns:
        RuntimeWarning: If a zero exhausted its insertion budget before the
            iterates stagnated, in which case the last iterate reached is
            reported and may be less accurate than ``tol`` asks for; and,
            separately, if a sign change exhausted that budget without its
            residual ever certifying it, in which case it is not reported at all
            and the result may be missing a root.

    Note:
        A value is reported only when the residual there is within what a zero
        may legitimately leave behind: the de Boor evaluation-error bound of the
        spline, plus ``|f'| * 2 * tol * scale``, the value the spline reaches at
        the parametric resolution the iteration stops at. The control polygon
        says where to look and never certifies on its own: the tracking of a sign
        change stops for three different reasons, and none of them implies that
        the point it stopped at is a zero. The residual test is also the only way
        a zero of even multiplicity is reported, since it does not change the
        sign of the spline and the sign changes bracketing it vanish under
        refinement.

        At a knot of multiplicity ``degree + 1`` the spline is discontinuous, and
        a zero of the piece on either side is reported at the knot. Evaluating
        there returns the *right* limit, so such a root can show a residual as
        large as the jump when the zero belongs to the left piece. Extracting the
        Bézier segments and solving each of them
        (:func:`pantr.bezier.find_roots`) reports the same knot for the same
        reason.

    Example:
        >>> import numpy as np
        >>> from pantr.bspline import BsplineSpace, BsplineSpace1D, Bspline, find_roots
        >>> knots = np.array([0.0, 0.0, 1.0, 2.0, 2.0])
        >>> space = BsplineSpace([BsplineSpace1D(knots, 1)])
        >>> spline = Bspline(space, np.array([[-1.0], [1.0], [3.0]]))
        >>> np.allclose(find_roots(spline), [0.5])
        True

    References:
        The unconditionally convergent knot-insertion method of Mørken and
        Reimers :cite:p:`morken2007zeros`.
    """
    return _find_roots_impl(bspline, tol=tol)


__all__ = ["find_roots"]
