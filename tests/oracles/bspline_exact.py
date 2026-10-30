"""Exact-rational B-spline oracle: the Cox-de Boor definition, summed over the control points.

The basis is evaluated from its recursive definition (not the triangle of Piegl and Tiller's
algorithms A2.2 and A2.3 that the library runs), over the whole knot vector, in
:class:`fractions.Fraction` arithmetic:

- ``N_{i,0}(x) = 1`` if ``i`` is the span of ``x`` and ``0`` otherwise;
- ``N_{i,q}(x) = (x - u_i) / (u_{i+q} - u_i) N_{i,q-1}(x)
  + (u_{i+q+1} - x) / (u_{i+q+1} - u_{i+1}) N_{i+1,q-1}(x)``, a term whose denominator is zero
  being zero;
- the ``k``-th derivative is the same recurrence differentiated ``k`` times by Leibniz's rule,
  ``D^k [(x - a) f] = (x - a) D^k f + k D^{k-1} f``, so no derivative formula of the library is
  reused.

Every knot and every point is a float, hence a dyadic rational: the oracle's values carry no
rounding at all, and a test compares against them with a tolerance that belongs to the library
alone. A periodic direction reads its control points modulo the number of stored ones (the
standard definition: the unwrapped knot vector's basis functions beyond the stored ones repeat
them). The tensor-product sum is taken over the local functions in exact arithmetic, and a
rational spline's derivatives come from the quotient rule applied to the exact homogeneous
derivatives.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from fractions import Fraction
from functools import lru_cache

import numpy as np
import numpy.typing as npt


def exact(x: float | np.floating) -> Fraction:
    """Get the exact value of a finite float or double as a fraction.

    Args:
        x (float | np.floating): A finite floating-point number.

    Returns:
        Fraction: Its value, without rounding.
    """
    return Fraction(float(x))


class ExactBasis:
    """The B-spline basis functions of one knot vector at one point, and their derivatives.

    Attributes:
        span (int): The knot span the point is evaluated on: the first ``j`` in
            ``[degree, len(knots) - degree - 2]`` with ``x < knots[j + 1]``, else the last (so the
            right end of the domain is evaluated on the last span, as the library does), or, with
            ``extend``, the first ``j`` in ``[0, len(knots) - 2]`` with ``knots[j] <= x <
            knots[j + 1]``: the periodic extension past the domain.
    """

    def __init__(
        self,
        knots: npt.ArrayLike,
        degree: int,
        x: float | np.floating,
        *,
        extend: bool = False,
    ) -> None:
        """Set up the basis at ``x``.

        Args:
            knots (npt.ArrayLike): The knot vector (floats; converted exactly).
            degree (int): The degree.
            x (float | np.floating): The point.
            extend (bool): Find the span among all intervals of the knot vector rather than the
                in-domain ones. Defaults to False.

        Raises:
            ValueError: If ``extend`` and ``x`` lies outside every interval of the knot vector.
        """
        self._u = [exact(k) for k in np.asarray(knots).tolist()]
        self._x = exact(x)
        self.degree = degree
        n = len(self._u)
        if extend:
            span = next(
                (j for j in range(n - 1) if self._u[j] <= self._x < self._u[j + 1]),
                None,
            )
            if span is None:
                raise ValueError("x lies outside the knot vector")
            self.span = span
        else:
            last = n - degree - 2
            self.span = next(
                (j for j in range(degree, last) if self._x < self._u[j + 1]),
                last,
            )
        self._d = lru_cache(maxsize=None)(self._derivative)

    def _derivative(self, i: int, q: int, k: int) -> Fraction:
        """Get ``D^k N_{i,q}`` at the point, on the polynomial piece of ``span``."""
        if q == 0:
            return Fraction(1 if (k == 0 and i == self.span) else 0)
        u, x = self._u, self._x
        result = Fraction(0)
        den1 = u[i + q] - u[i]
        if den1 != 0:
            term = (x - u[i]) * self._d(i, q - 1, k)
            if k > 0:
                term += k * self._d(i, q - 1, k - 1)
            result += term / den1
        den2 = u[i + q + 1] - u[i + 1]
        if den2 != 0:
            term = (u[i + q + 1] - x) * self._d(i + 1, q - 1, k)
            if k > 0:
                term -= k * self._d(i + 1, q - 1, k - 1)
            result += term / den2
        return result

    def value(self, i: int, k: int = 0) -> Fraction:
        """Get the ``k``-th derivative of the ``i``-th basis function at the point.

        Args:
            i (int): The function index in the (unwrapped) knot vector.
            k (int): The derivative order. Defaults to 0.

        Returns:
            Fraction: ``D^k N_{i,degree}(x)``.
        """
        return self._d(i, self.degree, k)

    def local(self, k: int = 0) -> list[Fraction]:
        """Get the ``degree + 1`` functions that can be non-zero on the span.

        Args:
            k (int): The derivative order. Defaults to 0.

        Returns:
            list[Fraction]: ``D^k N_{span - degree + j}`` for ``j = 0 .. degree``.
        """
        first = self.span - self.degree
        return [self.value(first + j, k) for j in range(self.degree + 1)]


def homogeneous(  # noqa: PLR0913
    knots: Sequence[npt.ArrayLike],
    degrees: Sequence[int],
    ctrl: npt.NDArray[np.floating],
    x: Sequence[float],
    orders: Sequence[int],
    *,
    extend: bool = False,
) -> list[Fraction]:
    """Evaluate a derivative of the sum over the control points, exactly (weights included).

    Args:
        knots (Sequence[npt.ArrayLike]): One knot vector per direction.
        degrees (Sequence[int]): The degree per direction.
        ctrl (npt.NDArray[np.floating]): Control points, ``(*num_stored, cp_size)``; a direction
            reads index ``i`` as ``i % num_stored`` (a no-op unless periodic).
        x (Sequence[float]): The point, one coordinate per direction.
        orders (Sequence[int]): The derivative order per direction.
        extend (bool): Evaluate on the periodic extension past the domain. Defaults to False.

    Returns:
        list[Fraction]: The ``cp_size`` coordinates of ``sum_j prod_d D^{k_d} N_{j_d}(x_d) c_j``.
    """
    dim = len(knots)
    bases = [ExactBasis(knots[d], degrees[d], x[d], extend=extend) for d in range(dim)]
    locals_ = [bases[d].local(orders[d]) for d in range(dim)]
    cp_size = ctrl.shape[-1]
    out = [Fraction(0)] * cp_size
    for js in itertools.product(*[range(degrees[d] + 1) for d in range(dim)]):
        w = Fraction(1)
        for d in range(dim):
            w *= locals_[d][js[d]]
        if w == 0:
            continue
        idx = tuple((bases[d].span - degrees[d] + js[d]) % ctrl.shape[d] for d in range(dim))
        row = ctrl[idx]
        for c in range(cp_size):
            out[c] += w * exact(row[c])
    return out


def derivative(  # noqa: PLR0913
    knots: Sequence[npt.ArrayLike],
    degrees: Sequence[int],
    ctrl: npt.NDArray[np.floating],
    x: Sequence[float],
    orders: Sequence[int] | None = None,
    *,
    rational: bool = False,
    extend: bool = False,
) -> list[Fraction]:
    """Evaluate a mixed partial derivative of the (projected) spline, exactly.

    A rational spline is the quotient of its first ``rank`` homogeneous coordinates by the last
    (the weight); its derivative follows the generalised quotient rule, ``R_k = (H_k -
    sum_{0 < i <= k} C(k, i) w_i R_{k - i}) / w_0``, over the multi-indices ``0 <= i <= k``
    ordered by total order.

    Args:
        knots (Sequence[npt.ArrayLike]): One knot vector per direction.
        degrees (Sequence[int]): The degree per direction.
        ctrl (npt.NDArray[np.floating]): Control points, ``(*num_stored, cp_size)``.
        x (Sequence[float]): The point.
        orders (Sequence[int] | None): The derivative order per direction. Defaults to None (the
            value).
        rational (bool): Whether the last coordinate is a weight. Defaults to False.
        extend (bool): Evaluate on the periodic extension past the domain. Defaults to False.

    Returns:
        list[Fraction]: The ``rank`` coordinates of the derivative.
    """
    dim = len(knots)
    ks = tuple(orders) if orders is not None else (0,) * dim
    if not rational:
        return homogeneous(knots, degrees, ctrl, x, ks, extend=extend)
    box = sorted(itertools.product(*[range(k + 1) for k in ks]), key=sum)
    hom = {i: homogeneous(knots, degrees, ctrl, x, i, extend=extend) for i in box}
    rank = ctrl.shape[-1] - 1
    w0 = hom[(0,) * dim][rank]
    result: dict[tuple[int, ...], list[Fraction]] = {}
    for k in box:
        v = list(hom[k][:rank])
        for i in box:
            if i == (0,) * dim:
                continue
            if sum(i) > sum(k):
                break
            if not all(i[d] <= k[d] for d in range(dim)):
                continue
            coeff = math.prod(math.comb(k[d], i[d]) for d in range(dim))
            r = result[tuple(k[d] - i[d] for d in range(dim))]
            for c in range(rank):
                v[c] -= coeff * hom[i][rank] * r[c]
        result[k] = [vc / w0 for vc in v]
    return result[ks]


def to_float_array(rows: Sequence[Sequence[Fraction]]) -> npt.NDArray[np.float64]:
    """Round exact rows to a ``float64`` array (each entry correctly rounded).

    Args:
        rows (Sequence[Sequence[Fraction]]): One row of exact values per point.

    Returns:
        npt.NDArray[np.float64]: The array, ``(n_rows, n_cols)``.
    """
    return np.array([[float(v) for v in row] for row in rows], dtype=np.float64)


def within(got: float | np.floating, want: Fraction, tol: float) -> bool:
    """Say whether a float is within ``tol`` of an exact value, compared exactly.

    Args:
        got (float | np.floating): The computed value.
        want (Fraction): The exact value.
        tol (float): The absolute tolerance.

    Returns:
        bool: Whether ``got`` is finite and ``|got - want| <= tol``.
    """
    if not math.isfinite(float(got)):
        return False
    return abs(exact(got) - want) <= exact(tol)
