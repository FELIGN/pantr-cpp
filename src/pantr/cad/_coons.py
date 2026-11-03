"""Coons patch and volume constructions.

Provides :func:`create_coons_surface` (bilinear blending from 4 boundary curves)
and :func:`create_coons_volume` (trilinear blending from 6 boundary faces).
"""

from __future__ import annotations

from .._ext import ext
from ..bspline import Bspline


def create_coons_surface(
    curves: tuple[tuple[Bspline, Bspline], tuple[Bspline, Bspline]],
) -> Bspline:
    """Construct a Coons patch from four boundary curves.

    Builds a bilinearly blended surface from four boundary curves using
    the formula ``S = R0 + R1 - B``, where *R0* and *R1* are ruled
    surfaces and *B* is the bilinear interpolant of the four corners.

    The curve layout is::

               C_u1
        o--------------o
        |  v           |
        |  ^           |
        |  |     C_v1  |
        |  +---> u     |
        o--------------o
        C_v0   C_u0

    **Rational curves are supported, and the blend is applied in homogeneous space.**  All
    three terms are formed from homogeneous coefficients ``(w x, w y, w z, w)`` and the
    result is projected only when it is evaluated, so the formula stays the linear one above
    and nothing has to be said about how weights combine under division.  That is what makes
    the boundary property survive: projection is pointwise, so it commutes with restriction
    to a boundary, and a blend whose *homogeneous* restriction to ``u = 0`` is ``C_v0``'s
    homogeneous data projects to ``C_v0`` itself.  A polynomial curve among rational ones is
    promoted with unit weights, so mixed input takes the same path rather than a third one.

    That argument has a hypothesis, and it is the one the corner check enforces: **the four
    curves must agree in homogeneous data where they meet, weights included, not merely
    projectively.**  A homogeneous representation is unique only up to one constant scaling
    all of a patch's weights, so two curves can name the same point of space at a shared
    corner while their coefficients differ by a factor, and the three-term inclusion-exclusion
    then does not cancel.  Curves that disagree are **refused** rather than blended, which is
    the verdict :func:`create_coons_volume` reaches on faces that disagree along a shared
    edge.  Rescaling *every* curve's homogeneous coefficients by one constant is not a
    disagreement and is accepted, since it describes the same patch.

    This is measured rather than feared: with a single rational boundary whose weights run 1
    to 2, every corner matched to ``1.1e-16`` and the patch missed one of the four curves it
    was handed by 17% of the model (pantr issue 327).

    Args:
        curves: ``((C_v0, C_v1), (C_u0, C_u1))`` -- two pairs of
            opposite boundary curves.  ``C_v0`` and ``C_v1`` run in the
            v-direction (left/right).  ``C_u0`` and ``C_u1`` run in the
            u-direction (bottom/top).  All must be 1D B-splines.

    Returns:
        Bspline: A 2D B-spline surface, in the precision the curves were given in: a
        float32 model is returned at float32 rather than promoted, matching
        :func:`~pantr.cad.create_ruled` and :func:`~pantr.cad.make_compat`.  Curves of
        *different* precisions are not a supported input and the result then follows
        ``curves[0][0]``'s.  Because it returns rather than raising, the four curves agree
        at every shared corner, and the patch therefore restricts to each of them on the
        corresponding boundary.  That interpolation is exact in exact arithmetic; the
        floating-point residual is **observed** to stay near ``1e-16`` relative, three
        orders below the tolerance the check below uses, and is not a proved bound.

    Raises:
        ValueError: If any curve is not 1D.
        ValueError: If two curves disagree at a shared corner by more than ``4096 * eps``
            times that column group's own magnitude over all eight corner readings: the
            largest absolute homogeneous coordinate for the coordinate columns, and the
            largest absolute weight for the weight column of a rational patch.  ``eps`` is
            read at the **coarsest** of the eight readings' precisions rather than at
            float64, so that a float32 model is graded against something float32 can
            express.  ``verify_corners_2d`` in ``include/pantr/cad/coons.hpp`` derives all
            of it.
    """
    (c_v0, c_v1), (c_u0, c_u1) = curves
    return Bspline._from_handle(ext.cad.create_coons_surface(c_v0._h, c_v1._h, c_u0._h, c_u1._h))


def create_coons_volume(
    faces: tuple[
        tuple[Bspline, Bspline],
        tuple[Bspline, Bspline],
        tuple[Bspline, Bspline],
    ],
) -> Bspline:
    """Construct a trivariate Coons blending volume from 6 boundary faces.

    Uses the trilinear Coons formula:

    ``V = (R_u + R_v + R_w) - (B_uv + B_uw + B_vw) + T``

    where *R* are ruled volumes from opposite face pairs, *B* are
    bilinear blend volumes from edge quadruples, and *T* is the
    trilinear corner interpolant.

    Edges and corners are derived automatically from face boundaries.

    The six faces are not independent data: each of the twelve edges is shared by two of
    them and each of the eight corners by three, so they have to agree where they meet.
    Faces that do not are refused rather than blended, because the formula spreads the
    inconsistency over the volume and the result then misses faces the caller supplied
    correctly.

    **Rational faces are supported, and the blend is applied in homogeneous space.**  All
    seven terms are formed from homogeneous coefficients ``(w x, w y, w z, w)`` and the
    result is projected only when it is evaluated, so the formula stays the linear one
    above and nothing has to be said about how weights combine under division.  That is
    what makes the boundary property survive: projection is pointwise, so it commutes with
    restriction to a face, and a blend whose *homogeneous* restriction to ``u = 0`` is
    ``face_u0``'s homogeneous data projects to ``face_u0`` itself.  A polynomial face
    among rational ones is promoted with unit weights, so mixed input takes the same path
    rather than a third one.

    That argument has a hypothesis, and it is the one the checks below enforce: the faces
    must agree in **homogeneous** data, weights included, not merely projectively.  A
    homogeneous representation is unique only up to one constant scaling all of a patch's
    weights, so two faces can trace the same edge in space while their coefficients differ
    by a factor, and then the inclusion-exclusion does not cancel.  This is measured rather
    than feared: scaling one ``w``-face's homogeneous control points by 2 leaves that
    face's geometry bitwise unchanged yet, with the checks disabled, makes the volume miss
    each of the four faces adjacent to it by 17% of the model.

    Face labelling convention:

    - ``faces[0] = (face_u0, face_u1)``: faces at u=0 and u=1,
      each a surface parameterized by (v, w).
    - ``faces[1] = (face_v0, face_v1)``: faces at v=0 and v=1,
      each a surface parameterized by (u, w).
    - ``faces[2] = (face_w0, face_w1)``: faces at w=0 and w=1,
      each a surface parameterized by (u, v).

    Args:
        faces: Three pairs of opposite boundary faces.

    Returns:
        Bspline: A 3D (trivariate) B-spline volume, in the precision the faces were given
        in: a float32 model is returned at float32 rather than promoted, matching
        :func:`~pantr.cad.create_ruled` and :func:`~pantr.cad.make_compat`.  Faces of
        *different* precisions are not a supported input and the result then follows
        ``faces[0][0]``'s.  Because it returns rather than raising,
        the six faces agree where they meet, and the volume therefore restricts to each of
        them on the corresponding boundary.  That interpolation is exact in exact arithmetic;
        the floating-point residual is **observed** to stay near ``1e-15`` relative, three
        orders below the tolerance the checks below use, and is not a proved bound.

    Raises:
        ValueError: If any face is not a 2D B-spline.
        ValueError: If two faces meeting at a corner of the volume disagree there by more
            than ``4096 * eps`` times the largest absolute coordinate over all
            twenty-four corner readings, derived in ``verify_corners_3d`` of
            ``include/pantr/cad/coons.hpp``.  Same tier as
            :func:`create_coons_surface`, but one undivided scale rather than that
            function's two column groups: these are **projected** points and so carry one
            physical dimension between them, while a patch compares homogeneous
            coefficients.  A weight disagreement on the volume path is caught by
            ``verify_edges_3d`` instead, which does split the columns.
        ValueError: If two faces sharing an edge disagree along it by more than
            ``4096 * eps`` times that column group's own magnitude over the twelve
            compared edges: the largest absolute homogeneous coordinate for the
            coordinate columns, and the largest absolute weight for the weight column of
            a rational edge, derived in ``verify_edges_3d`` of
            ``include/pantr/cad/coons.hpp``.  Both tolerances read ``eps`` at the coarsest of
            the compared readings' precisions rather than at float64.
        ValueError: If the blend of rational faces carries a control weight that is not
            above zero, so that the weight field of the result cannot be certified
            positive.  The bound is ``w(u) >= min_i w_i`` and is sufficient rather than
            necessary, so this refuses some volumes whose weight field never vanishes;
            ``verify_positive_weights`` in ``include/pantr/cad/coons.hpp`` derives it.
    """
    (face_u0, face_u1), (face_v0, face_v1), (face_w0, face_w1) = faces
    return Bspline._from_handle(
        ext.cad.create_coons_volume(
            face_u0._h, face_u1._h, face_v0._h, face_v1._h, face_w0._h, face_w1._h
        )
    )
