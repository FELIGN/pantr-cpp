# NaN and infinity

PaNTr has one policy for non-finite numbers (NaN and ±infinity): **propagate, never hide.**

- Non-finite data in gives non-finite data out, at every value that reads it. A value that does
  not read the non-finite entry stays finite: a B-spline value reads only the control points of
  its knot span, and a Bézier evaluated at `t = 1` is its last control point.
- No function returns a finite value computed from a non-finite entry it reads, or reports success
  (a root count, a convergence, a rank, a residual, a passed validation) on non-finite data.
- A non-finite value that is born from finite input, by an internal overflow, is an error: it
  raises, and does not reach you as a NaN. The one NaN a function returns from finite input is
  {func}`~pantr.bezier.find_monotone_root`'s "no root" sentinel, which is documented and not a
  failure (see "Root finding refuses" below).
- The fast paths add no validation for this. A function that evaluates at many points does not
  scan its inputs first; it lets the arithmetic carry the NaN to the values that read it.

The rest of this page says how each family of functions applies the policy. Every example is run
as shown, and the output is what it printed (with warnings turned into errors, so a silent
`RuntimeWarning` would have stopped it).

## Evaluation propagates

A NaN or infinite control point appears in the values at the points whose evaluation reads it.

```python
import numpy as np
from pantr.bspline import Bspline, BsplineSpace, BsplineSpace1D

space = BsplineSpace([BsplineSpace1D([0, 0, 0, 1, 2, 3, 3, 3], 2)])
curve = Bspline(space, np.array([[0.0], [1.0], [np.nan], [3.0], [4.0]]))
print(curve.evaluate(np.array([0.0, 0.5, 1.5, 3.0])).ravel())
```

```text
[nan nan nan nan]
```

A B-spline value depends on the `degree + 1` basis functions of the knot span that holds the
point. A control point outside that span does not enter the value, so it does not make it NaN:

```python
curve = Bspline(space, np.array([[np.nan], [1.0], [2.0], [3.0], [4.0]]))
print(curve.evaluate(np.array([0.5, 1.5, 3.0])).ravel())
```

```text
[nan  2.  4.]
```

A Bézier curve carries an infinite control point the same way, with one asymmetry between its
ends. The 1-D evaluation sums each control point times its Bernstein value, and at `t = 0` the
Bernstein values of all but the first control point are exactly zero. A zero times an infinity or a
NaN is NaN, so a non-finite control point other than the first makes the value at `t = 0` NaN,
although its weight there is zero. At `t = 1` the evaluation does not sum: it returns the last
control point, so the value is the last control point whatever the others hold, finite or not:

```python
from pantr.bezier import Bezier

bezier = Bezier(np.array([0.0, np.inf, 1.0]))
print(bezier.evaluate(np.array([0.0, 0.25, 0.5, 1.0])))

# The infinity is now the last control point: `t = 1` returns it, and `t = 0` is still NaN.
print(Bezier(np.array([0.0, 1.0, np.inf])).evaluate(np.array([0.0, 1.0])))
# A non-finite *first* control point is the value at `t = 0`, and `t = 1` does not read it.
print(Bezier(np.array([np.inf, 0.0, 1.0])).evaluate(np.array([0.0, 1.0])))
```

```text
[nan inf inf  1.]
[nan inf]
[inf  1.]
```

The finite value at `t = 1` is therefore the last control point alone and not a guarantee that
the curve is finite: the same curve is NaN at `t = 0`.

A NaN *parameter* is not hidden either. A Bézier returns NaN there, and a B-spline refuses it as
outside the domain:

```python
print(Bezier(np.array([0.0, 1.0, 2.0])).evaluate(np.array([np.nan, 0.5])))

good = Bspline(space, np.array([[0.0], [1.0], [2.0], [3.0], [4.0]]))
try:
    good.evaluate(np.array([np.nan, 1.0]))
except ValueError as e:
    print(type(e).__name__, e)
```

```text
[nan  1.]
ValueError One or more values in pts are outside the knot vector domain (0.0, 3.0)
```

A product of Bézier curves carries a NaN control point to the product control points that read it,
and a product of *finite* control points whose result overflows raises (the third rule above):

```python
a = Bezier(np.array([1.0, 2.0, 3.0]))
nan = Bezier(np.array([1.0, np.nan, 3.0]))
print(np.asarray(a.multiply(nan).control_points).ravel())

big = Bezier(np.array([1e200, 1e200, 1e200]))
try:
    big.multiply(big)
except OverflowError as e:
    print(type(e).__name__, e)
```

```text
[ 1. nan nan nan  9.]
OverflowError The product of these finite control points overflows the range of its dtype, or an intermediate of the Bernstein product formula overflows float64
```

## Root finding refuses

A root set for a polynomial with a NaN coefficient has no meaning: "no roots" would be a
statement about a polynomial that does not exist, and a root would be a guess.
{func}`~pantr.bezier.find_roots` and {func}`~pantr.bezier.find_monotone_root` therefore raise
`ValueError` for a NaN or infinite coefficient (single or batch; a batch names the first bad row),
and for a `tol` that is not positive and finite:

```python
from pantr.bezier import find_monotone_root, find_roots

for cp in ([0.0, np.nan, 1.0], [-1.0, np.inf, 1.0]):
    try:
        find_roots(Bezier(np.array(cp)))
    except ValueError as e:
        print(type(e).__name__, e)

b = Bezier(np.array([-1.0, 0.0, 1.0]))
for tol in (np.nan, np.inf):
    try:
        find_roots(b, tol=tol)
    except ValueError as e:
        print(type(e).__name__, e)
    try:
        find_monotone_root(b, tol=tol)
    except ValueError as e:
        print(type(e).__name__, e)
print(find_roots(b))
```

```text
ValueError coefficients must be finite, got a NaN or an infinite entry
ValueError coefficients must be finite, got a NaN or an infinite entry
ValueError tol must be positive and finite, got nan
ValueError tol must be positive and finite, got nan
ValueError tol must be positive and finite, got inf
ValueError tol must be positive and finite, got inf
[0.5]
```

{func}`pantr.bspline.find_roots` refuses a NaN or infinite `tol` in the same way (it raised
`ValueError` ("tol must be positive and finite, got nan") for both in a run of its own).

One NaN that {func}`~pantr.bezier.find_monotone_root` returns is not a failure. For *finite*
coefficients with no sign change across `[0, 1]`, there is no root to return, and NaN is its
documented "no root" sentinel (in batch mode, one entry per polynomial, NaN where none exists).
It is a statement about a polynomial that exists and has no root there, which is the reverse of
the NaN-coefficient case above, so it raises nothing and a caller tests for it with `np.isnan`:

```python
print(find_monotone_root(Bezier(np.array([1.0, 2.0, 3.0]))))
print(find_monotone_root([Bezier(np.array([-1.0, 0.0, 1.0])), Bezier(np.array([1.0, 2.0, 3.0]))]))
```

```text
nan
[0.5 nan]
```

## Fits and interface detection refuse a non-finite `tol`

`tol` is the truncation tolerance of the least-squares solves behind
{func}`~pantr.bspline.fit_bspline`, {func}`~pantr.bspline.interpolate_bspline`,
{func}`~pantr.bspline.l2_project_bspline`, `fit_bezier` and `interpolate_bezier` (and of the
distributed fits in {mod}`pantr.mpi`), and the match tolerance of
{func}`~pantr.multipatch.detect_interfaces`. A NaN or infinite value used to make every
singular value count as zero, so the fit returned all-zero control points as if it had succeeded,
or matched no faces, or every face. It raises now, before the callable of `l2_project_bspline`
is sampled:

```python
from pantr.bspline import create_uniform_space, fit_bspline, l2_project_bspline

space = create_uniform_space(2, 4)
x = np.linspace(0.0, 1.0, 9)
y = x**2
for tol in (np.nan, np.inf):
    try:
        fit_bspline(y, x, space, tol=tol)
    except ValueError as e:
        print("fit_bspline", type(e).__name__, e)
try:
    l2_project_bspline(lambda lat: lat.pts_per_dir[0] ** 2, space, tol=np.nan)
except ValueError as e:
    print("l2_project_bspline", type(e).__name__, e)
```

```text
fit_bspline ValueError tol must be finite, got nan
fit_bspline ValueError tol must be finite, got inf
l2_project_bspline ValueError tol must be finite, got nan
```

```python
from pantr.bspline import Bspline
from pantr.multipatch import detect_interfaces

cp = np.linspace(0.0, 1.0, space.num_total_basis)[:, None]
try:
    detect_interfaces([Bspline(space, cp)], tol=float("nan"))
except ValueError as e:
    print(type(e).__name__, e)
```

```text
ValueError tol must be finite, got nan
```

A zero `tol` is accepted (`fit_bspline(x**2, x, space, tol=0.0)` returned finite control points),
as is a negative one. Non-finite *data* passed to a fit gives non-finite control points:

```python
y[3] = np.nan
print(np.asarray(fit_bspline(y, x, space).control_points).ravel())
```

```text
[nan nan nan nan nan nan]
```

## Knot factories refuse a non-finite domain

A NaN or infinite domain end would give a knot vector of NaN or infinite knots, which the next
function to read it would meet far from the cause.
{func}`~pantr.bspline.create_uniform_open_knots` and
{func}`~pantr.bspline.create_uniform_periodic_knots` raise `ValueError` instead:

```python
from pantr.bspline import create_uniform_open_knots, create_uniform_periodic_knots

for f in (create_uniform_open_knots, create_uniform_periodic_knots):
    for domain in ((0.0, float("nan")), (0.0, float("inf"))):
        try:
            f(4, 2, domain=domain)
        except ValueError as e:
            print(f.__name__, domain, type(e).__name__, e)
```

```text
create_uniform_open_knots (0.0, nan) ValueError domain must be finite, got [0.0, nan]
create_uniform_open_knots (0.0, inf) ValueError domain must be finite, got [0.0, inf]
create_uniform_periodic_knots (0.0, nan) ValueError domain must be finite, got [0.0, nan]
create_uniform_periodic_knots (0.0, inf) ValueError domain must be finite, got [0.0, inf]
```

## Overflow from finite input raises

Refining a hierarchical space computes a two-scale matrix between levels in the knots' own
floating-point type. On a `float32` knot vector whose span is wider than the `float32` range, a
ratio of knot differences is infinity over infinity. {meth}`pantr.bspline.THBSplineSpace.refine`
and {meth}`~pantr.bspline.THBSplineSpace.refine_region` raise `OverflowError` rather than build a
space whose prolongation holds NaN:

```python
from pantr.bspline import BsplineSpace1D, create_thb_space

v = np.float32(2e38)
knots = np.array([-v, -v, -v, 0.0, v, v, v], dtype=np.float32)
coarse = create_thb_space(BsplineSpace([BsplineSpace1D(knots, 2)]))
try:
    coarse.refine_region(0, [0], [2])
except OverflowError as e:
    print(type(e).__name__, e)
```

```text
OverflowError the two-scale matrix between two levels overflows the knots' dtype (a knot span beyond its range)
```
