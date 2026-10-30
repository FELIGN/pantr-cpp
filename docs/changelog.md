# Changelog

## Unreleased

### Added
- **`pantr.bezier.set_root_provider` and `get_root_provider`** select which root finder
  `find_roots` and `find_monotone_root` use: `"builtin"` (pantr's own) or `"berf"` (BeRF-FR,
  from the `pantr-berf` package). The initial choice comes from the `PANTR_ROOT_PROVIDER`
  environment variable, read on first use (an unknown name raises `ValueError` then, not at
  import); without it, `"berf"` when `pantr-berf` is installed, else `"builtin"`. Asking for
  `"berf"` without the package raises `ImportError`. This is the runtime switch for the
  certified finder the C++ port adds as an optional package.

### Rejected where it used to be accepted
- **`num_intervals=0` is refused by all three knot-vector factories.**
  `create_uniform_open_knots` and `create_uniform_periodic_knots` documented
  `num_intervals` as *"must be non-negative"* and their shared validator enforced exactly
  that, so zero was inside the stated contract and neither returned a usable vector.
  `create_uniform_periodic_knots(0, 3)` returned `[nan, nan, nan, 0.0, inf, inf, inf]` --
  the periodic extension divides by the interval count -- announcing itself only through
  a `RuntimeWarning` that nothing raises on, so the caller received a mostly non-finite
  knot vector and met it somewhere downstream. `create_uniform_open_knots(0, 3)` returned
  a finite vector with *one* interval rather than none, which constructs a perfectly
  ordinary space and is therefore the quieter of the two. `create_cardinal_knots` had
  always refused zero with `num_intervals must be at least 1`, so the module already
  carried the right answer one function away; the shared validator now applies that rule
  to the two that call it, and both docstrings say so. `create_cardinal_knots` keeps its
  own copy of the check rather than delegating, because it derives its domain from the
  count and a zero would reach the validator as `domain=(0, 0)` and be reported against a
  parameter the caller never passed. This is the same rule `BsplineSpace1D` applies to a knot
  vector whose domain is a single point, moved one step earlier to where the offending
  argument still has a name: the constructor could only report that the vector it was
  handed spans no interval, which for the periodic case was not even the reason it
  failed.
- **A knot vector whose last knot repeats more than `degree + 1` times is refused by
  `BsplineSpace1D`** at positive degree. It used to build a space whose basis does not sum
  to 1 at the right endpoint. An excess at the first knot, an interior repeat and degree 0
  are still accepted.
- **`Bspline.elevate_degree` refuses a direction that is not clamped** (a run of
  `degree + 1` equal knots at each end) unless it is periodic, which is converted to open
  form first. It used to fail on a coefficient count or return a different function.
  `Bspline.to_open_bspline` is the way forward.
- **`Bspline.derivative(keep_degree=True)` refuses an unclamped direction**, and so does a
  rational `derivative()` in either setting, since both re-elevate. It used to return a
  function that disagrees with the true derivative. A non-rational `derivative()` with
  `keep_degree=False`, and a direction of degree 1, are unaffected.
- **A rational `derivative()` on a periodic direction with a C^0 seam is refused**
  (seam multiplicity equal to the degree). The derivative is discontinuous at the seam and
  has no periodic form; the error says so. Before, the same input raised a misleading
  "The B-spline is not periodic" error.
- **`reduce_degree` on a rational `Bezier` or `Bspline` refuses a reduction that makes a
  control weight zero or negative.** The result would have a pole on the domain and would
  not be a NURBS. `Bezier.minimize_degree` already declined such reductions.
- **The extraction dispatchers refuse an identity-flagged operator that is not square.**
  The identity branch of the kernels assumes a square operator and used to read past the
  operand. No public path builds such an operator; direct callers of the Layer 2 helpers
  were exposed.
- **Complex `PointsLattice` coordinates are refused** (`ValueError`), and a non-integer
  point count given to `create_lagrange_points_lattice` raises `TypeError` instead of being
  truncated. A complex lattice used to keep its dtype, which no tabulation could use.
- **An unknown Lagrange variant is refused** by `tabulate_lagrange_1d`, `tabulate_lagrange`,
  `create_lagrange_points_lattice` and `compute_lagrange_to_bernstein_1d` with a `ValueError`
  (`'...' is not a valid LagrangeVariant`). Any value that was not one of the five used to select
  the Chebyshev second-kind nodes silently. The variant is now converted once, for the C++
  library, which knows the five families and nothing else; except in `tabulate_lagrange`, which
  still checks its degrees first, it is checked before the count or degree
  (`compute_bernstein_to_lagrange_1d` already refused it).

### Changed
- **pantr needs Python 3.12 or later.** pantr is becoming a C++ library with a compiled
  Python extension, built for CPython's stable ABI so that one wheel per platform serves
  every Python from 3.12 on; the binding library's stable-ABI support starts at 3.12.
  Python 3.11 is no longer supported.
- **pantr is built with scikit-build-core and ships compiled extension modules.** A source
  install needs a C++20 compiler and CMake 3.28 or later. In a source checkout, `make ext`
  builds the extension into `src/pantr` (there is no editable install), and importing pantr
  fails with a message saying so when the module is missing or was built from other
  bindings.
- `PointsLattice` copies the arrays it is given and freezes the copies, so a later change
  to the caller's arrays no longer alters the lattice. Its arrays are read-only, and
  `pts_per_dir` returns fresh read-only views on each access. Code that wrote through
  `lattice.pts_per_dir[0][:] = ...` now raises `ValueError`.
- `Bspline.derivative(keep_degree=True)` on a B-spline with several directions converts
  only the differentiated direction to open form. The other directions keep their
  periodicity and their knot vectors; before, every direction came back open and clamped.
- `to_periodic` grades the seam in control-point units, from the control points' scale and
  dtype. The budget no longer depends on the knot vector's magnitude or on the parametric
  `tol`.
- Every Layer 2 entry point over a `parallel=True` kernel now waits for the JIT warmup
  thread before its first call, which avoids a process abort when a call races the warmup.
- **Quadrature, tolerance, basis tabulation and change of basis run in pantr's C++
  library** (`pantr.quad`, `pantr.tolerance`, `pantr.basis`, `pantr.change_basis` are thin
  wrappers over it; names, signatures and defaults are unchanged, and messages too except as
  listed below). Floating-point results can move in the last bits, and an ill-conditioned
  change-of-basis matrix by up to the accuracy bound its builder documents (past degree 10 in
  float64 for the cardinal-to-Bernstein one, a bound above 100%): Gauss-Legendre and Gauss-Lobatto nodes come from
  Newton's iteration on the Legendre recurrence instead of NumPy's eigenvalue route, the
  tanh-sinh step from a Lambert W evaluation of pantr's own, and the change-of-basis
  matrices from Eigen's LU instead of LAPACK's. `compute_monomial_to_bernstein_1d` is
  unchanged to the bit, every entry the double nearest `C(i, j) / C(degree, j)`.
- **The Lagrange basis is evaluated by pantr's barycentric formula**, with weights
  accumulated as mantissa and exponent so they neither overflow nor underflow, instead of
  SciPy's `BarycentricInterpolator` and its seeded random permutation of the products. The
  values are the same polynomial to the accuracy of the formula, reproducible on every
  platform and with every SciPy version; they are not bit-identical to the previous
  release's. The 16-epsilon snap to the Kronecker row at a node is kept.
- **The basis tabulations run on pantr's own thread pool and release the GIL** while a
  large batch is computed, so several Python threads can tabulate at once. Results do not
  depend on the thread count. `pantr.set_num_threads` sets their parallelism.
- **`PointsLattice` stores coordinates of a dtype other than float32 and float64 as
  float64.** The lattice is a C++ value of one of those two types; an integer lattice used
  to keep its dtype, and no tabulation could use it.
- **Masked point arrays are refused by the Lagrange tabulations at every degree >= 1, first**:
  the previous release refused them only on reaching SciPy (after the degree, variant and `out`
  checks); the C++ kernel would otherwise tabulate the masked-out values as data.
- **`PointsLattice` and `QuadratureRule` pickle and copy through their constructor**
  (`__reduce__`), since they now hold a C++ value that Python cannot pickle by itself. A
  `copy.deepcopy` of either is therefore immutable like the original: its arrays are read-only
  (a deep copy used to give writable arrays).
- `PointsLattice.get_all_points("C")` returns a C-contiguous array (it was a transposed,
  F-contiguous view); the values are unchanged. The C++ lattice builds the points in the order
  asked for, so no transpose is left to return.
- **Results are NumPy arrays that do not own their data**: a tabulation, rule or matrix is a C++
  buffer handed to NumPy without a copy (`flags.owndata` is `False`), so `ndarray.resize` raises
  on it where it used to work; copy the array first.
- **Dtype arguments in another byte order** (`">f8"`, `np.dtype(">f4")`) give native-order
  results; the C++ library computes in native `float` or `double`, and the previous release cast
  to the requested order. `compute_lagrange_to_bernstein_1d` and
  `compute_bernstein_to_lagrange_1d` now accept such a dtype, which they refused with
  "Output array has dtype ...".
- **Integer-like counts and degrees are accepted** by the 1D rules, `create_lagrange_points_lattice`
  and the change-of-basis builders: an object with `__index__` (or a 0-d integer array) is
  converted as an integer by the compiled binding, where the previous release raised `TypeError`
  or NumPy's "Series must have maximum degree of at least 1". The 1D and tensor tabulations and
  `gauss_legendre_quadrature` refuse them as before (a `float` `ndim` there is now the binding's
  `TypeError`; it used to be accepted).
- **No `RuntimeWarning` from the tabulations** at infinite or subnormal points: the values are
  the same, computed in C++ rather than through NumPy operations that warned.
- **A `PointsLattice` of strings is refused** ("could not convert string to float"); it used to
  be built with a string dtype no tabulation could use (non-float coordinates are converted to
  float64, see above).
- **Counts and degrees outside the annotated types or past memory** may be refused with another
  exception or message than before, because the C++ library's binding and allocation now see
  them first. In the 1D rules, `create_lagrange_points_lattice`, the 1D tabulations and the
  change-of-basis builders, a count or degree that is not an integer (a float, even `2.0`, a
  string, `None`) or an integer outside the int64 range is a `TypeError` from the compiled
  binding; the previous release raised assorted `ValueError`, `OverflowError`, `IndexError`,
  `MemoryError` and `TypeError`s, and some rules accepted it (a float count in the Lobatto and
  Chebyshev rules, `2**63` and `2**63 - 1` in the two Chebyshev ones). A `bool` is accepted as
  the integer it is. A degree above `2**31 - 1`, the C++ `int`, is a `ValueError` ("exceeds the
  largest supported degree") in the 1D and tensor tabulations and the builders; a
  change-of-basis matrix too large to allocate is a `ValueError` ("... more elements than fit in
  memory") raised before any work; an allocation that fails in C++ is a `MemoryError` reading
  "std::bad_alloc" (a 1D tabulation's output is still allocated by NumPy, with NumPy's text); and a rule of `2**62` points is a `ValueError` naming the C++ container's
  limit where NumPy raised "array is too big" or a `MemoryError`. `gauss_legendre_quadrature`
  still converts its counts (not `ndim`) with `int()` and checks `ndim` first (a `bool` `ndim` is now quoted
  as `1` in its message), and the tensor tabulations still refuse a non-integer degree with "All
  degrees must be non-negative integers". The 1D tabulations convert their points before
  checking the degree, so a call with both invalid reports the points.
- **An invalid `funcs_order`** in the tensor tabulations is a `ValueError` ("funcs_order must be
  'C' or 'F'") checked first. The previous release treated it as `'C'` for an array of points and
  raised NumPy's `einsum` error for a `PointsLattice`; an unchecked value is a silent ordering.
- `tensor_product_quadrature` accepts any iterable of `(nodes, weights)` pairs (an iterator
  used to raise `TypeError` for want of `len`): the pairs are converted once, in order, for the
  C++ call. So an empty iterator is "rules must have at least one axis" and a non-iterable a
  `TypeError` reading "... is not iterable" rather than "... has no len()".
- **Which of two invalid arguments is reported** follows the wrappers' order: the Lagrange
  variant first (before points, dtype, `out` and counts, except the tensor `tabulate_lagrange`,
  which checks its degrees first), then the dtype, then the points, then sizes; a lattice's
  complex coordinates are reported before its empty ones. A non-integer degree in the 1D
  tabulations reads "'<=' not supported" (the wrapper's range check) where it read "'<' not
  supported".
- **A parallel tabulation can fail to start pool threads** under a tight address-space limit
  (`ulimit -v`): the call raises `RuntimeError` ("Resource temporarily unavailable"), and
  `pantr.set_num_threads(1)` runs it on the calling thread.
- When both the size (`n_pts`, `degree` or a lattice's counts) and `dtype` are invalid, the
  quadrature functions and the change-of-basis builders report the `dtype` first, because the
  wrapper selects the `float` or `double` C++ function before calling it; each is reported as
  before when it is the only invalid argument. The private imports `pantr.quad` and
  `pantr.change_basis` used to re-export by accident (NumPy's `chebyshev` and `legendre`,
  `functools`, `comb`, a few pantr functions; `Callable` from `pantr.change_basis`) are gone from
  their namespaces.
- **`AABB`, `AffineTransform` and the `Bezier` value type run in pantr's C++ library**
  (`pantr.geometry`, `pantr.transform` and `pantr.bezier.Bezier` are thin wrappers over it;
  names, signatures and defaults are unchanged, and messages too except as listed below).
  Results can move in the last bits: Bézier contractions are summed point by point in C++ instead of by
  `einsum`/`tensordot` (on a lattice, in whichever order of directions writes fewer values),
  the 1D evaluation seeds its recurrence with an exact integer power instead of `pow`, a
  polynomial curve's derivative is evaluated on its hodograph (the forward differences of its
  control points, a Bézier of lower degree) instead of the derivative basis, which is several
  times faster, and affine maps and transforms are applied by plain loops instead of BLAS.
  Evaluation, and `AffineTransform.__call__` on many points, run on pantr's thread pool with the
  GIL released (NumPy's `@` ran on BLAS's threads). Where a result overflows, the plain loops can
  leave NaN where BLAS left `inf` (or the reverse), and NumPy's `RuntimeWarning`s for overflow and
  invalid values in these operations are no longer emitted. The sign of a zero result can differ (the values
  compare equal): the `-0.0` against `+0.0` tie in `AABB.union` and `intersect`, a zero bound
  from `AABB.transform`, a zero entry of `AffineTransform.inverse`'s offset.
- A derivative of a polynomial `Bezier` above its degree in some direction is computed as the
  exact zero it is, without the derivative table, at any order: an order of `2**31 - 1` or more is
  refused as "derivative order N is too large" (`ValueError`), where the table used to be
  allocated and fail with a `MemoryError` past memory. At a point with a NaN or infinite
  coordinate it is NaN; the previous release gave NaN only when such a coordinate lay in a
  direction whose derivative order was below its degree, and zero otherwise.
- At a point with an infinite coordinate, derivatives within the degree can be NaN where the
  previous release gave `inf` or `-inf` (or the reverse), since the evaluation order differs.
- **A `Bezier` is immutable**, as the C++ values are (they are shared rather than copied, which
  is what makes them cheap to pass and safe across threads): the constructor copies the control
  points, so a later write to the array it was given no longer changes the Bézier;
  `control_points` is a read-only view; `reverse()` returns an independent Bézier (it shared
  memory with the original through `np.flip`); `reverse`, `permute_directions` and `transform`
  with `in_place=True` replace the value the object holds, so a `control_points` array read
  before the call keeps the values it had. `Bezier.to_bspline(copy=False)` gives a B-spline
  holding that read-only view, which cannot then be changed in place. `create_from_bspline` and
  `Bspline.to_bezier` with `copy=False` still share the B-spline's array (read-only through the
  Bézier) when it is a C-contiguous float32 or float64 array, as before; an array of another
  layout (a reversed view, say) or dtype is converted, so the Bézier no longer shares it.
- **`Bezier` keeps `float32` and stores every other real dtype as `float64`**: integers as
  before, and now booleans, `float16`, `longdouble` and byte-swapped arrays (`">f4"` becomes
  native `float32`). The previous release stored those dtypes as given and then failed in
  evaluation (or, for booleans, computed in boolean arithmetic). Complex control points are a
  `TypeError`. The C++ library instantiates the two storage types only.
- **`Bezier`, `AABB` and `AffineTransform` pickle and copy explicitly** (`__reduce__`; for
  `Bezier` also `__copy__` and `__deepcopy__`, which share the immutable value), since they now
  hold a C++ value Python cannot pickle by itself. A deep copy is immutable like the original:
  its arrays are read-only, where they used to be writable. A `Bezier` or `AffineTransform`
  pickled by the previous release still loads: its plain attributes rebuild the value.
- `AABB.lo` and `hi`, `AffineTransform.matrix` and `offset`, `Bezier.control_points` and the
  Bézier results are NumPy arrays over C++ buffers that do not own their data
  (`flags.owndata` is `False`); the read-only ones cannot be made writeable
  (`flags.writeable = True` raises `ValueError`, where `AABB.lo` and the matrix of an
  `AffineTransform` used to allow it, being arrays the object owned). `Bezier.control_points` is a
  cached property and `AABB` has an extra slot; assigning or deleting `control_points` is
  refused as before.
- `AffineTransform.inverse` refuses a matrix whose LU factorisation meets a NaN pivot
  ("Cannot invert a singular affine transformation."), as it refuses an exactly zero one;
  NumPy returned NaNs there or raised, depending on where the NaN sat. One rule for the C++
  library, NumPy's for the zero pivot. Which numerically singular matrices meet an exactly zero
  pivot also changes, since Eigen's elimination rounds differently from LAPACK's:
  `[[1, 2, 3], [4, 5, 6], [7, 8, 9]]` is now inverted (to entries of order `1e15`, noise, as any
  inverse of it is) where it was refused, and some other near-singular matrices are refused where
  they were inverted. A matrix with a zero row or a zero column is refused as before. A matrix
  with equal rows is refused up to 16 dimensions (the previous release inverted some of them to
  noise, depending on BLAS rounding); above that size neither library refuses them reliably. Results can move in the last bits for a well-conditioned matrix;
  for an ill-conditioned one they move within the bound the condition number gives.
- `AABB.transform` given an offset without `ndim` entries is a `ValueError` ("affine.offset must
  have shape ..."). NumPy broadcast a one-entry or 0-d offset to every axis and raised a
  broadcasting error for other lengths; the C++ library checks every input.
- **Inputs outside the annotated types, and calls with two invalid arguments**, in
  `pantr.geometry`, `pantr.transform` and `Bezier`, may raise another exception or message than
  before, since the compiled binding converts an argument (an index, a float) before the checks
  run and most checks now run in C++: for example a `bool` direction can be quoted as `1`, a
  non-numeric `factor` or `angle` is a `TypeError` from the binding before any range check, an
  integer outside the int64 range for a `direction`, `axis`, `ndim`, `n`, `dim`, `component` or
  degree increment is a `TypeError` from the binding (it was a `ValueError` from the range
  check), and a `center` of the wrong rank in the factories reports its ravelled shape. A `bool`
  given for an integer is the integer it is (`True` is `1`) everywhere: `AABB.empty(True)`,
  `AABB.unbounded(True)`, `AffineTransform.identity(True)` and `Bezier.permute_directions` with
  `bool` entries are accepted where they were a `TypeError`; `AffineTransform.shear` with a `bool`
  `component` or `direction` (`True` or `False`) shears along that axis (the previous release used
  it as a mask); `Bezier.evaluate_derivatives(points, True)` and a list of `bool` orders are the
  first derivative; the previous release raised `ValueError` for a surface or volume and, for a
  curve, returned the table of orders 0 and 1 with an extra axis. `False` is order 0 (it raised
  `IndexError` for a curve). Messages can still echo a `bool` as given. Bounds and split
  values given as NumPy scalars are converted to `double` by the binding, whatever their
  precision: `restrict` with two `np.float32` bounds on a `float64` Bézier computes in `float64`
  (it computed in `float32`, because both bounds kept their `float32` type).
- **Under a tight address-space limit** (`ulimit -v`), the first parallel `Bezier.evaluate` or
  `AffineTransform.__call__` can fail to start pool threads, as the tabulations can (above):
  `RuntimeError` ("Resource temporarily unavailable") or `MemoryError` ("std::bad_alloc"), and
  `pantr.set_num_threads(1)` runs it on the calling thread.
- The private modules `pantr.bezier._bezier_core`, `_bezier_eval`, `_bezier_derivative`,
  `_bezier_compose`, `_bezier_collapse`, `_bezier_split`, `_bezier_slice` and `_bezier_restrict`,
  and `_bezier_degree._degree_elevate_bezier`, are gone with the Numba kernels they held, as are
  the private helpers `pantr.geometry._BOUNDS_NDIM`, `_BOUNDS_AXIS_LEN` and `_require_same_ndim`,
  `pantr.transform._apply_center` and `_with_optional_center`, the helpers
  `pantr.bezier._bezier_product._multiply_bezier`, `_multiply_nonrational`, `_multiply_rational`,
  `_ensure_rational_ctrl` and `_append_unit_weight_column`,
  `_bezier_utils._tabulate_Bernstein_basis_1D_core`,
  `_bezier_degree._degree_elevate_bezier_1d_core` and `_tabulate_Bernstein_basis_1D_serial_core`,
  and the attribute `Bezier._control_points`, whose work is now done in C++.

- **Degree reduction (`Bezier.reduce_degree`, `degree_reduction_error`, `minimize_degree`),
  fitting (`interpolate_bezier`, `fit_bezier`) and root finding (`find_roots`,
  `find_monotone_root`) run in pantr's C++ library**; `pantr.bezier` is now a thin wrapper
  over it, with the names, signatures and defaults it had. The reduction operators are
  assembled in exact rationals as before (now Boost.Multiprecision) and are the same bits.
  `reduce_degree` and `find_monotone_root` reproduce the previous release's values bit for bit
  on `float32` and `float64`, and so does `find_roots` except where Bezier clipping used to drop
  roots (under "Fixed") and in the depth-cap case below (measured over random, constructed and
  special-value inputs). `degree_reduction_error` sums its Gram contraction in another order and
  moves within that contraction's rounding, which grows with the degree: from the last bits at
  low degree, to about `1e-6` relative at degree 20 for random coefficients (`1e-1` for smooth
  ones), `1e-3` at degree 25 and tens of percent and more near degree 30, where the
  previous release's own value was as far from the exact one; its docstring no longer calls the
  value exact. `minimize_degree` rests its decisions on the same contraction: it makes the
  previous release's decisions up to degree about 23, and above that its final degree can differ
  (neither is reliably the exact greedy answer there). `degree_reduction_error` and
  `minimize_degree` no longer emit NumPy's `RuntimeWarning`s (invalid value, overflow) on
  non-finite or overflowing control points; the results are the same. The fits' singular value
  decomposition is Eigen's instead of LAPACK's, so a fit moves within the bound the Vandermonde
  matrix's conditioning gives (a few units in the last place on a small well-conditioned grid,
  growing with the node count and the conditioning). Batch root finding runs on pantr's thread
  pool with the GIL released. Like the other C++ results, the arrays `find_roots` and
  `find_monotone_root` return do not own their data (`ndarray.resize` refuses them), except the
  empty results of `find_roots([])` and `find_monotone_root([])`.
- **`find_roots` warns when Bézier clipping gives up on part of `[0, 1]`** (`RuntimeWarning`,
  "... exhausted the root finder's subdivision budget ..."). Clipping abandons an interval that
  is still wider than `tol` after 64 levels of subdivision; it used to do so silently. It is
  rare: it needs a polynomial whose clips are exact (such as some straight lines written at a
  higher degree) and a `tol` below the spacing of doubles at its root, so that no interval can narrow to
  `tol`; whether a run of clips that each remove little can reach it as well is not established.
  A root found there can differ from the previous release's by a unit in the last place.
- **`fit_bezier` and `interpolate_bezier` with `float16`, `longdouble` or byte-swapped values
  compute in `float64`** (nodes included) and return a `float64` Bézier (also for byte-swapped
  `float32`, which `Bezier` itself would store as native `float32`). `fit_bezier` failed in the previous release's Numba basis kernel
  (`NotImplementedError` for `float16`, a Numba `TypingError` for `longdouble` and byte-swapped
  arrays) whenever a direction had more than one node, and returned the values' dtype with one
  node per direction. `interpolate_bezier` returned `longdouble` for `longdouble` values, and the
  values' dtype with one node per direction; it now returns `float64` in both cases.
- **`reduce_degree`, `degree_reduction_error`, `minimize_degree`, `find_roots` and
  `find_monotone_root` accept a Bézier built from `float16`, `longdouble`, boolean or
  byte-swapped control points**, which `Bezier` now stores as `float64` (or native `float32`).
  For `float16`, `longdouble` and byte-swapped arrays the previous release failed in its kernels
  (`NotImplementedError`, a Numba `TypingError`), and root finding raised `TypeError` ("... must
  be a float32 or float64 ndarray ..."). For boolean control points its `reduce_degree` returned a
  boolean Bézier whose every point was `True` (silently wrong), `degree_reduction_error` raised
  NumPy's boolean-subtract `TypeError` and `minimize_degree` a `ValueError` ("... not inexact");
  all now compute on the `float64` values.
- **A NaN or infinite node in `fit_bezier` or `interpolate_bezier`** (or a scattered point), in a
  direction with more than one node, raises `ValueError` ("SVD did not converge"), as before, but
  no longer `numpy.linalg.LinAlgError` (a `ValueError` subclass), since the decomposition is no
  longer NumPy's.
- **A negative `degree` for scattered `fit_bezier` nodes** is a `ValueError` ("degree must be
  non-negative"); the previous release raised `IndexError` for `-1` and NumPy's "negative
  dimensions are not allowed" below it. An empty sequence of node arrays is a `ValueError`
  ("fit_tensor: values need ... one node array ... per direction") for scalar or 1-D values; the
  previous release returned a Bézier. An empty node array is a `ValueError` ("bernstein_pinv needs 0 <=
  degree < len(nodes) ..."); it was an `IndexError`.
- **Inputs outside the annotated types** of `fit_bezier`, `interpolate_bezier`, `minimize_degree`,
  `find_roots` and `find_monotone_root`, as for `Bezier` above, can be accepted where they were
  refused or refused with another exception and message, since the binding converts `tol` to a
  `double`: a NumPy `float16`, `longdouble` or `bool` scalar, a 0-d array, a `Fraction` or a
  `Decimal` `tol` is accepted (the previous release failed in Numba for several of them in the
  root finders). A one-element array `tol`, accepted before by `fit_bezier`, `interpolate_bezier`,
  `minimize_degree` and (for one polynomial) the root finders, is now a `TypeError`; a list or a
  string was and is a `TypeError`, with a new message in the first three.
- The private modules `pantr.bezier._bezier_degree`, `_bezier_utils`, `_root_finding_core`,
  `_yuksel_core`, `_clipping_core` and `_batch_core`, and the helpers
  `_bezier_interpolate._bernstein_interpolate`, `_bernstein_vandermonde_svd`,
  `_build_bernstein_pinv`, `_build_nd_bernstein_vandermonde`, `_fit_from_scattered`,
  `_fit_from_values`, `_find_roots._validate_coeff_array`, `_dispatch_single`,
  `_CLIP_MIN_DEGREE`, `_CLIP_COEFF_RANGE_LIMIT` and `_SUPPORTED_DTYPES`, are gone with the Numba
  kernels and NumPy code they held, whose work is now done in C++. The B-spline reduction's
  weight check moved to `pantr.bspline._bspline_degree`.
- **`BsplineSpace1D`, `BsplineSpace`, the knot factories (`create_uniform_open_knots`,
  `create_uniform_periodic_knots`, `create_cardinal_knots`, `create_uniform_space`) and
  `get_greville_abscissae` run in pantr's C++ library**; names, signatures, defaults and messages
  are unchanged except as listed below. The knot vectors, knot classes, tolerances, interval
  counts, first-basis indices, cardinal-interval flags, windowed spaces (`restrict`) and Greville
  abscissae are the previous release's to the bit. Basis values and derivatives can move in the
  last bits, since the C++ build fuses a multiply and an add where the previous one rounded twice:
  within a few units in the last place of the type the recurrence runs in, which is the points'
  type for the values and the knots' type for the derivatives (a `float32` space's derivatives at
  `float64` points are computed in `float32`, as before).
- **The basis tabulations of `BsplineSpace1D` and `BsplineSpace` run on pantr's thread pool and
  release the GIL** for a large batch; results do not depend on the thread count, and
  `pantr.set_num_threads` sets their parallelism. The serial kernel the previous release used for
  fewer than 4096 points is gone (the pool decides from the work per point).
- **`BsplineSpace1D` and `BsplineSpace` pickle and copy explicitly** (`__reduce__`): a space
  rebuilds from its stored knots and tolerance, so the copy equals the original, and a pickle of
  the previous release still loads. `copy.copy` and `copy.deepcopy` of a `BsplineSpace1D` share
  its immutable C++ value.
- **A masked knot array** gives a `BsplineSpace1D` whose `knots` is a plain read-only array of the
  data (it used to be a masked array; the mask was never applied).
- **`degree=True` or `False` is the integer it is** (`1` or `0`) in `BsplineSpace1D`; the previous
  release failed in Numba's typing (`TypingError`) on most knot vectors.
- **A `float32` point array on a `float64` Bézier-like space** (`BsplineSpace1D.tabulate_basis`,
  `tabulate_basis_derivatives`), and points of a dtype other than the space's in a one-direction
  `BsplineSpace.tabulate_basis`, are accepted and tabulated in the points' dtype (respectively the
  space's): the previous release refused them with "Output array has dtype ...". A list of points
  (or any array-like) is accepted by `BsplineSpace.tabulate_basis`, which raised `AttributeError`.
- **No `RuntimeWarning` from `create_uniform_open_knots` or `create_uniform_periodic_knots`** at an
  infinite domain end; the knot vectors are the same.
- **Inputs outside the annotated types** in `BsplineSpace1D` and the knot factories raise another
  exception or message than before, since the compiled binding converts a degree, a count or a
  continuity to an integer before any check runs: a `float` degree in `BsplineSpace1D` is a
  `TypeError` from the binding (it was Numba's `TypingError`), and a `float` count, degree or
  continuity in a factory is a `TypeError` with the binding's message instead of NumPy's.
- The private module `pantr.bspline._bspline_basis_multidim`, and the helpers
  `_bspline_basis_core._tabulate_Bspline_basis_1D_impl`, `_tabulate_Bspline_basis_deriv_1D_impl`,
  the Bernstein-like and serial kernels and `_PARALLEL_MIN_NUM_PTS`,
  `_bspline_knots._knot_scale`, `_knot_tolerance`, `_KNOT_MERGE_SAFETY`,
  `_check_snapping_kept_an_interval`, `_check_space_has_an_interval`, `_validate_knot_input`,
  `_get_last_knot_smaller_equal_impl` and `_bspline_space_1d._cached_unique_knots_and_multiplicity`
  are gone with the Numba kernels and NumPy code they held, whose work is now done in C++.

### Fixed
- **`find_roots` no longer loses roots that Bezier clipping found.** Clipping collected its
  candidates in a fixed buffer of `3 * degree + 4` entries and silently dropped any past it, which
  polynomials with many exact, evenly spaced roots fill: at degree 7 with roots `1/8, ..., 7/8`,
  the previous release returned six of the seven (the same at degrees 11, 15, 23, 31 and for
  other dyadic families). The candidate list now grows.
- `AABB` can be pickled and copied (`pickle`, `copy.copy`, `copy.deepcopy`); each raised
  `AttributeError` from the immutability guard, which refused the state the default protocol
  restores.
- `AffineTransform(matrix)`, `AffineTransform.translation` and the other factories no longer make
  the caller's own float64 array read-only (the constructor set `writeable = False` on it when it
  needed no conversion); the C++ value holds its own copy.
- Multi-dimensional tabulations at zero points in three or more dimensions return an empty
  `(0, n_functions)` array; they raised a NumPy reshape error.
- `get_modified_chebyshev_nodes_1d` honors `dtype` when `n_pts` is a NumPy integer; it returned
  float64.
- High-order Bernstein derivatives (reached by `Bezier.evaluate_derivatives` and the
  Bezier-like B-spline paths) are now correct where the scaling `degree! / (degree - k)!`
  passes 2^63, for example from order 14 at degree 30. The scaling
  was an `int64` that wrapped; it now switches to double precision when it would overflow.
- `create_lagrange_points_lattice` given a one-shot iterable of counts (a generator) now
  builds the lattice it describes. The counts were read twice: the validation exhausted the
  generator, and the construction then saw no direction and raised "Points lattice must
  have at least 1 dimension".
- `reduce_degree` on a periodic B-spline now succeeds for a smooth spline. The closure
  step used to compare the reduction's own approximation error against round-off and
  refused nearly every case.
- `derivative()` of a periodic rational B-spline now works, for both values of
  `keep_degree`. It used to raise "Inserting these knots would exceed the maximum
  multiplicity" from deep inside the knot insertion.
- An `out` array in Fortran order (or any layout whose reshape is a copy) is now written.
  The basis and basis-derivative functions used to fill a copy and leave the caller's
  array as allocated.
- Bezier root finding no longer loses roots to underflow. Sign tests compare signs instead
  of multiplying the two values, which flushed to zero for tiny coefficients, and the
  test for a vanishing coefficient no longer carries an absolute floor, so rescaling a
  polynomial no longer changes its roots.

## 0.7.0 (2026-08-19)

### Added
- `pantr.multipatch`: new subpackage for multi-patch topology. `Interface` and
  `MultiPatch` describe how patches meet; `detect_interfaces` finds the
  interfaces of a patch collection by enumerating candidate orientations and
  verifying them, which resolves the ambiguity a corner-matching rule leaves on
  symmetric faces; `match_face_cps` pairs the control points across an
  interface. Knots, coordinates and NURBS weights are compared against separate
  tolerances, since they are not the same kind of quantity.
- `pantr.bspline.find_roots`: every zero of a scalar univariate B-spline,
  computed on the spline's own knot vector by repeatedly inserting the zero of
  the control polygon (Mørken & Reimers, *Math. Comp.* 76, 2007). No Bézier
  extraction, no stitching of roots at segment boundaries, and about 8 times
  faster than extraction plus the Bernstein solver on a cubic with 1000
  elements.
- `Bspline.locate`: point inversion from physical to parametric coordinates,
  batched over all points, returning the cell and the parameters. The default
  tolerance scales with the coordinate magnitude, not only with the bounding-box
  diagonal, so a geometry far from the origin does not silently lose points, and it
  carries what the *parametrization* can resolve as well: one ulp of a parametric
  coordinate becomes a physical residual of about `eps * |xi|`, so a knot span far
  from the origin relative to its width was being held to a residual no representable
  parameter could reach, and every query on such a patch came back not found. That
  term grades acceptance only: the iteration stops at the image-only length, so the
  coordinate returned is as accurate as the geometry allows rather than as accurate
  as the bar it is certified against (see *Changed* below).
- `BsplineSpace.boundary_dofs`: the control-point indices of a boundary slab of
  chosen thickness, on a chosen face.
- `BsplineSpace.cell_supports` and `BsplineSpace1D.first_basis_per_interval`:
  which basis functions are supported on which cells, and the index of the first
  one per knot interval. Both are cached and computed in exact integer
  arithmetic, with no basis evaluation.
- `THBSplineSpace.prolongation_to_sparse`: the CSR counterpart of
  `prolongation_to`, assembled column by column so the dense operator is never
  built. Storage grows linearly with the coarse space where the dense form grows
  quadratically (measured over a refinement sweep: 12.9x against 120.6x).
- `THBSplineSpace.max_active_per_cell`, `HierarchicalGrid.export_cells` and
  `Grid.boundary_facets`: the largest number of active functions on any cell, a
  deduplicated vertex/cell export for downstream meshes, and the boundary facets
  of a grid, with an axis-aligned default on the base class.
- `SpanwiseElementExtraction.factors`: per-cell access to the univariate
  extraction factors, as views sharing memory with the stored operators.
- `pantr.change_basis`: `compute_legendre_to_cardinal_1d`,
  `compute_cardinal_to_legendre_1d` and
  `compute_cardinal_dual_legendre_coeffs_1d`, completing the Legendre-cardinal
  pair. The transforms are obtained from an LU solve on the cardinal matrix
  rather than from its Gram matrix, which would square the conditioning; the
  docstrings state the accuracy actually attainable per degree instead of
  promising a fixed bound.
- `pantr.bspline.dof_owner_windowed`: owner lookup restricted to a window of
  degrees of freedom, for the distributed layer.
- `Bezier.degree_reduction_error`: the exact $L^2$ norm of the error
  `Bezier.reduce_degree` would introduce, computed through the Bernstein mass
  matrix rather than by sampling.
- `HierarchicalGrid.coarsen_cells`: the cell-exact counterpart of `coarsen`. It groups the
  ids by parent and demotes a parent only when every one of its children is both an active
  leaf and named, so it can never destroy a cell the caller did not list, which the box
  `coarsen` can after two overlapping refines. The box form stays, and stays cheaper for a
  region known to be uniformly refined. `THBSplineSpace.coarsen` now runs on it and keeps
  only its admissibility guard, which has to stay a per-parent loop: evaluating it against
  the frozen input mesh instead of the evolving one changes the result on 94 of 600
  randomized meshes.
- `HierarchicalGrid.cell_id`: the flat id of an active leaf given its level and
  multi-index, or `None`. The inverse of `cell_level` and `cell_multi_index`, and how a
  caller re-resolves cells after a refine or a coarsen has reassigned ids.

### Rejected where it used to be accepted
- **Four Coons boundary curves that disagree in homogeneous data at a shared corner are
  refused.** The patch's corner check read its corners through `boundary()`, which
  *projects*, so both readings were physical points and agreed exactly whenever the
  geometry did; the weight never entered the comparison. But a homogeneous representation
  is unique only up to one constant scaling all of a patch's weights, so two curves can
  name the same point of space at a corner while supplying different `(w x, w y, w z, w)`
  there, and the three-term blend — formed homogeneously and projected only on evaluation
  — then fails to cancel. Measured on a unit square with a single rational boundary whose
  weights run 1 to 2: every corner matched to `1.1e-16` and the patch missed one of the
  four curves it had been handed by `1.7e-01`, 17% of the model. Corners are now compared
  as homogeneous coefficients, with the coordinate and weight columns in separate scale
  groups so a dimensionless weight is never graded against the model's length — the rule
  `_verify_shared_boundary` and the volume's edge check already apply to the same kind of
  comparison — and a disagreement raises a `ValueError` naming the corner and the column
  group that decided. Rescaling *every* curve's homogeneous coefficients by one constant
  is not a disagreement and is still accepted; it describes the same patch. Polynomial
  input is unaffected.
- **A weight disagreement between two rational Coons faces is refused at every model
  size.** The edge check took one magnitude over a homogeneous control row, which mixes
  coordinates (length times weight) with weights (dimensionless), so a weight error was
  graded against the model's *length* and passed unnoticed once the model was large: a
  weight off by four times the tolerance was accepted at every coordinate scale from 1e2
  up, and the volume then missed faces the caller had supplied correctly. Control-point
  columns are now split into groups that each share a physical dimension and each graded
  against their own magnitude. The rule lives in one place, shared with `join`.
- **A knot vector whose spacing is finer than its dtype resolves at its own coordinate
  magnitude is now refused at construction**, where it used to be accepted and then
  silently behave as if it had no intervals. `BsplineSpace1D` raises a `ValueError`
  naming the dtype, the coordinate magnitude, the merge tolerance in ulp and the closest
  pair of knots you supplied, and suggesting float64, a domain nearer the origin, or a
  coarser mesh. Concretely: float32 on `[1e6, 1e6 + 1]` with more than one interval. The
  ulp there is 0.0625, so a four-interval mesh spaces breakpoints 0.25 apart while
  telling two knots apart at that magnitude needs 0.48 — an interior knot computed by any
  route is uncertain by 6% of the window, and no tolerance both keeps the mesh and merges
  two routes to the same knot. The previous release kept the mesh only by never merging
  anything (its tolerance there was 1.6e-6 ulp), so the knots it stored were noise.
  Reachable in float64 too, but only from `|offset| / span ≈ 1.4e14` at four intervals,
  where the window is 8 ulp wide. Two things pass *this* check: a knot vector that was
  already a single repeated value, which is not this check's cause and gets the message
  below instead, and anything built with `snap_knots=False`, which bypasses the merging
  and this diagnosis together. Neither escapes the interval requirement below, which
  `snap_knots=False` could previously be used to evade.
- **A knot vector whose in-domain knots all fall in one class is refused at
  construction.** `BsplineSpace1D` accepted it, and each consumer then met the
  degeneracy on its own terms: `tensor_product_grid` named the breakpoints,
  `Bspline.locate` surfaced numpy's zero-size `min`, `tabulate_basis_derivatives`
  divided by zero, and `Bspline.evaluate`, `tabulate_basis` and `evaluate_derivatives`
  returned `0.0` or `NaN` without raising at all. Those last three are why the rule
  lives in the constructor rather than in each entry point: they return a value, so a
  survey of what *raises* would not have found them. The predicate is the interval
  count, not "every knot is the same value", because the family is wider than it looks:
  `[0, 1, 1, 1, 2]` at degree 1 holds three distinct values and still has a domain of
  `(1.0, 1.0)`, the domain running from `knots[degree]` to `knots[-degree-1]`. Counting
  intervals catches that case together with the flat one and adds no threshold of its
  own, the count being taken with the tolerance the space already uses everywhere else.
  `create_cardinal_knots` has always applied the same rule, as `num_intervals must be at
  least 1`; the direct constructor was the hole. The `ValueError` names both domain ends
  and the tolerance, and reports a *consecutive step* rather than the span between the
  ends: knot grouping splits on `knots[i] - knots[i-1] > tol` and therefore chains, so at
  float32 around 1e6, where the tolerance is 0.954, three knots 0.5 apart in turn merge
  although the ends are 1.0 apart, and a message claiming the ends are within tolerance
  would have been false there.

### Changed
- **`pantr.tolerance`'s presets are now dimensionless relative tolerances.**
  `get_strict`, `get_default` and `get_conservative` return `K * eps(dtype)` with
  `K` equal to 4, 64 and 4096 — an acknowledged safety factor, stated, a power of
  two, and the *same* in every precision. The previous table was a fixed value per
  dtype, and in units of that dtype's own epsilon it was neither constant nor
  ordered the same way: strict was 0.10 eps in float16, 0.84 in float32 and 4.5 in
  float64, so `get_strict` demanded bitwise equality in the two smaller formats
  while allowing four roundings in the largest. The module docstring called the
  presets absolute while `Bspline.locate` and both root finders were already
  multiplying them by a magnitude; that ambiguity is what this change removes. A
  preset is now unusable on its own — multiply it by the magnitude of the quantity
  being compared, and name that magnitude at the call site. Numerically:
  `get_default(float64)` moves from 1e-12 to 1.42e-14 and `get_conservative`
  from 1e-10 to 9.09e-13, while `get_strict(float32)` moves from 1e-7 to 4.77e-7.
  The per-dtype table and its platform branch on whether `longdouble` aliases
  `float64` are gone: reading the epsilon from the platform makes both unnecessary.
- **`BsplineSpace1D.tolerance` now carries the knot vector's own magnitude.** It is
  `8 * eps(dtype) * max(span, |knots[0]|, |knots[-1]|)` — the strict preset,
  doubled for one extra rounding upstream, scaled by the parametric magnitude every
  comparison in the B-spline layer is relative to. Previously it was the bare
  per-dtype constant, so the same absolute band meant four ulp on a unit domain,
  nothing at all from `|knot| ≈ 5` upward, and half the breakpoints of a
  twenty-interval float32 mesh on `[0, 1e-6]`, which then reported ten. Everything downstream that takes `space.tolerance` — domain membership,
  knot matching, cardinal interval lengths, knot insertion and removal, products,
  degree changes — becomes scale-covariant with it. `detect_interfaces` is the one
  caller that wanted the dimensionless factor instead and now takes it from
  `pantr.tolerance` directly, since none of the three quantities it compares is a
  parametric coordinate of a single patch.
- **The individual sites that still carried a bare constant now derive one.** The
  presets above made a tolerance mean something; these are the places that could not be
  re-derived until it did. Each grades a named quantity, scales by that quantity's own
  magnitude, and states its safety factor in an attribute docstring:
  - **The THB refinement gate** certified a candidate whenever the worst prolongation
    residual stayed under `1e-8 * (1 + max_coarse_value)`. The shape was right — the
    residual is compared against the size of the coarse column it reproduces — but the
    constant was wrong by seven decades, so an impostor whose residual landed anywhere in
    `[1e-14, 1e-8]` was certified and `prolongation_to` returned a matrix that does not
    prolong. It is now `4096 * eps * (1 + max_coarse_value)`, the conservative tier
    standing in for the backward-stability constant of the `gelsd` least-squares solve,
    which has no closed form. Measured over 132 configurations (1D/2D/3D, degrees 2–5,
    truncated and not, one and two refinement levels, three domain scales) the worst
    residual is 18.5 eps, so the gate keeps a safety factor of 443 while tightening
    1.1e4-fold. A `sqrt(M)` growth term was tried and dropped on measurement: `M` spans
    13 to 19683 with no trend in the residual.
  - **THB cell membership** allowed a fixed `1e-12` of slack at a cell boundary, which
    rejected a point one ulp outside once the parametric domain reached 1e6 and accepted
    some 9000 ulp of slop on `[0, 1]`. It is now `8 * eps * max(|lo|, |hi|, hi - lo)` per
    axis: the same rule, and the same number, the knot layer uses for the same question.
  - **The Coons corner check** decided a `ValueError` with `atol=1e-12, rtol=0`. Across
    twelve decades that rejects a one-ulp corner mismatch at model size 1e6 — a
    micron-unit model of a metre-scale part failing on a geometrically perfect patch —
    while accepting a 1e-9 *relative* gap at 1e-6. It is now `4096 * eps` times the
    largest absolute coordinate over **all eight** corner values, so the verdict does not
    depend on which corner sits at the origin.
  - **`Bspline.locate`'s geometric scale** no longer floors at 1.0. The floor held a
    millimetre-scale part to a thousand times the relative accuracy of a metre-scale one;
    the fallback to 1.0 now fires only for a geometry with no extent *at the origin*,
    which is the one case with no scale to read. The docstring's account of the default
    tier is corrected with it: the `60 * eps * scale` it cited as a six percent margin was
    a censored measurement, since the iteration stops at the threshold. Re-measured at six
    thresholds from 2 to 4096 epsilons over 9480 query points, the achieved residual
    tracks whatever threshold is set and nothing is lost even at `4 * eps * scale`, so the
    threshold is an accuracy contract rather than a safety margin, and loosening the tier
    would cost accuracy one for one.
  - **`multipatch`'s joint geometric tolerance** was `relative * diagonal`, which tracks
    how big the geometry is but not where it sits: two unit patches meeting at 1e6 were
    graded at 2.5e-18, eight orders below their own noise floor. It now takes
    `max(diagonal, largest |coordinate|)`.
  - **`Bspline.restrict`** gains no constant, which is its policy and is now written
    down: every comparison it makes is a knot-identity question that
    `BsplineSpace1D.tolerance` already answers, and the module docstring records the three
    properties making one number enough — insertion preserves the endpoints, the
    extraction offsets widen away from the interval, and the snapping invariant stops the
    widening over-collecting.
- **`Bspline.locate` stops at the accuracy it owes and accepts at what the map can
  attain.** One number used to do both jobs: the iteration halted as soon as the residual
  passed it, so the coordinate returned was only ever as accurate as the bar it was
  graded against. That cost nothing while the bar was small, but once the threshold
  gained the parametric-resolution term above, geometry parametrized far from the origin
  earns a legitimately larger bar and stopped early against it, on queries whose exact
  preimage is representable and whose accuracy was there for the taking. The two
  questions are now separate, and only one of them grows with the parametrization: the
  iteration stops at `get_default(dtype)` times the physical length, which is the
  accuracy contract, while a point is accepted at that same length maximized against
  what one ulp of a parametric coordinate moves the image, which is an attainability
  statement, so a query whose exact answer is unrepresentable is still found instead of
  lost. Neither introduces a constant. Measured on a warped unit patch whose second
  direction is parametrized on `[1e6, 1e6 + 1]`, 60 queries: the worst residual falls
  from 1.20e-08 to 4.99e-11 for targets that are exact images, and from 1.19e-08 to
  1.88e-10 for targets half an ulp off one, with none of the 60 lost either way, against
  4 of 60 and 60 of 60 lost by the tight threshold used alone. Ordinary geometry pays
  nothing for it: the two thresholds are the same float bit for bit for any knot vector
  spanning `[0, L]`, and a 2400-query sweep measures 1.000x map evaluations at physical
  offsets 0 and 1e6. An explicit `tol` still governs both, so a caller asking a proximity
  question is not charged for accuracy they did not ask for.
- **`Bspline.remove_knots` is now lossless by default.** `tol=None` used to mean an
  absolute `1e-10`, a geometric budget wearing no units: the same exactly-removable knot
  came out at geometry scale 1e-6, 1 and 1e3 and was silently refused at 1e6 and 1e9. It
  now means *remove only what is exact to round-off*, `8 * (degree + 1) * eps *
  max(bbox diagonal, largest |P_i|) * sqrt(n)`, which is the one default needing no
  arbitrary constant. Measured over 105 cases (degrees 1–7, ranks 1–3, geometry scales
  1e-6 to 1e9) by bisecting for the smallest budget that still removes a just-inserted
  knot: the largest distance observed is 0.84 eps × scale, and the tightest margin
  anywhere in the set is a factor of 48. The `sqrt(n)` is what the *other* directions
  add: on a surface or a volume the removal test is handed a slice with every other
  direction's control points flattened into it, so its single Euclidean distance spans
  `n` control points rather than one, and stacking `n` blocks each of norm at most `e`
  reaches `sqrt(n) · e`. Without that term the default silently refuses an exactly
  redundant knot once `n` is large enough — measured on a degree-3 volume, from
  `n = 124609` upward. **This is a behaviour change** — a knot whose removal costs up to
  1e-10 of geometry used to come out and no longer does. Trading geometry for a shorter
  knot vector is a price only the caller can set, so it has to be asked for.
- **A rational knot-removal budget is converted before it reaches the kernel.** The kernel
  measures a Euclidean distance over every column of the array it is given, which for a
  NURBS is the homogeneous `[w·x, w·y, w·z, w]`, while the caller's budget is a distance
  in projected space. Eq. (5.30) of Piegl & Tiller (2nd ed., 1997, p. 185),
  `TOL = d · w_min / (1 + |P|_max)`, is now applied. Without it a deviation carried by the
  **weight** column is graded as though it were a coordinate: sweeping a weight
  perturbation over sixteen decades with the coordinates at 1e6, the worst projected
  deviation an accepted removal produced was 7.4e5 times the requested budget; with the
  conversion, 0.32. A circle cannot show this, its weights being structurally tied to its
  coordinates, so the tests vary weights and coordinates independently.
- `pantr.grid.overlay` merges breakpoints closer than the default *relative*
  tolerance times the axis's own magnitude -- the intersection window and the
  coordinates bounding it -- instead of against a bare `float64` constant. A
  window of length 1 sitting at 1e6 resolves its breakpoints no better than
  `eps * 1e6` however short it is, and the merge verdict is now the same on
  `[0, 1]` as on any rescaling of it.
- The minimum supported SciPy is now 1.15, raised from 1.11. Lagrange tabulation
  seeds the node permutation SciPy's `BarycentricInterpolator` applies, and the
  `rng` argument that makes that possible arrived in 1.15. Earlier versions
  cannot be made reproducible through that class at all — 1.11 accepts neither a
  seed nor precomputed weights — so the previous floor was already untrue for
  this function rather than merely untested.
- Knot insertion runs the Oslo recurrence over the `degree + 1` band each row is
  supported on, instead of over the full row. Refining a spline with 10⁴ control
  points dyadically takes 1.9 ms where it took 1432 ms. The refinement matrix
  returned by the internal dense entry point is unchanged bit for bit, so the
  THB, spline-product and periodic-conversion paths are untouched; inserted
  control points sum the same products in a different order and can move in the
  last bit.
- `dof_owner` and `compute_halo` are compiled. Measured on 10⁶ control points in
  3D at degree 3: 68 ms and 168 ms. They stay single-threaded on purpose — the
  Numba workqueue threading layer is not threadsafe against the warmup thread
  the package starts at import.
- The span search shared by the basis and derivative kernels is fused into the
  parallel loop instead of running as a serial pass before it. It dominated the
  runtime, so multi-core speed-up was capped near 1.3x by Amdahl's law.
- `Bezier.reduce_degree` and `Bspline.reduce_degree` now interpolate the
  endpoints of every segment: among the lower-degree polynomials that reproduce
  the original at both ends of the parametric domain, the result is the closest
  in $L^2$. Reduced control points therefore differ from previous releases on
  any reduction that is not exact. In exchange, adjacent B-spline segments meet
  at breakpoints bit for bit and the previous averaging of the shared control
  point — which moved *both* segments off their own optima — is gone. Without
  the endpoint conditions the $L^2$ error would be a factor 1.1 (degree 16) to
  4.5 (reduction to a straight line) smaller. Reduction to degree 0 keeps no
  endpoint condition and still returns the mean of the control points.

### Fixed
- **A rational Coons patch carries the corner weight its curves agreed on.** The corner
  term of `create_coons_surface` was read through `boundary()` and then promoted with unit
  weights, so four curves agreeing homogeneously at every corner still produced a patch
  that missed them whenever the weight they agreed on was not 1: measured `2.2e-02` on a
  degree-2 model whose corner weights were all 2, with nothing refused, and the blend's
  corner weight came out as `2w - 1` instead of `w`. It is now read homogeneously and built
  by the same helper the volume's corner term uses (`_build_trilinear_volume` became
  `_build_multilinear_corner_term`, taking its dimension from the corners), so the two
  paths cannot drift apart again. Polynomial patches at float64 are byte-identical to
  before. At float32 they are byte-identical too *unless* the corner term needs degree
  elevation or knot insertion to meet the two ruled terms, in which case they differ by a
  couple of float32 ulps (measured: 2 ulps, `2.384e-07` at unit scale): the corner term
  used to be built at float64 whatever the model's dtype and rounded down once at the end,
  and it is now built in the curves' own precision like every other term. That is the
  no-promotion rule the previous release adopted for this function, applied to the one term
  that still escaped it.
- **A Coons patch from curves carrying two coordinates builds** instead of dying in NumPy
  with `operands could not be broadcast together with shapes (2,2,2) (2,2,3) (2,2,2)`,
  which named nothing the caller had supplied. The corner term no longer goes through
  `create_bilinear`, a public *geometric* primitive that zero-pads its corners to rank 3 —
  the same reason the volume never used `create_trilinear` for its own corner term.
- **`create_coons_volume` builds from rational faces** instead of failing with a NumPy
  shape error. The seven-term blend was already formed from homogeneous coefficients
  throughout; only the corner read left that space, through `boundary()`, which projects.
  Corners are now read homogeneously. The boundary property survives projection because
  projection is pointwise and therefore commutes with restriction to a face, provided the
  faces agree in *homogeneous* data, weights included, rather than merely projectively. A
  rational volume is additionally refused when its weight field is not certified positive:
  the blend is an inclusion-exclusion and subtracts three of its terms, so positive face
  weights do not imply a positive weight field inside, while `w(u) >= min_i w_i` over a
  non-negative partition of unity lets the result's own control weights bound it from
  below. That test is sufficient and not necessary, and says so where it refuses.
- **The Coons boundary checks are graded at the input's own precision.** The tier was read
  at float64 whatever the model's dtype, so a float32 patch whose corner is as close as
  the format allows was refused for a disagreement float32 cannot express, by a factor of
  `eps32 / (4096 eps64) = 2**17`. `create_coons_surface` also stops promoting a float32
  model to float64, matching `create_ruled` and `make_compat`.
- Knot snapping merged the wrong knots, and from `|knot| ≈ 5` upward merged none
  at all. `BsplineSpace1D._snap_knots` rounded onto a grid of width `tolerance`,
  which was an absolute per-dtype constant: at knot 1.0 the grid was 4.5 ulp wide
  and it narrowed relative to the knots as they grew, so two knots one ulp apart
  stopped merging at base 5 and the detected multiplicity dropped — a cubic asked
  for multiplicity 2 (C¹) silently became C². A rounding *grid* is also wrong in
  principle: two values one ulp apart can straddle a grid line and never merge,
  for any spacing. Knots are now grouped by relative gap — a class ends where the
  step to the next knot exceeds `8 * eps * max(span, |knots[0]|, |knots[-1]|)`,
  three roundings of an affine map of the span with 2.7x over it — and every
  member of a class takes the class's *first* knot. Choosing rather than averaging
  makes snapping idempotent and keeps the stored values ones the caller supplied;
  `np.mean` over `degree + 1` identical copies is not even the identity at large
  magnitude, and it was moving the reported domain (`1000001.0` came back as
  `1000000.875` in float32).
- `get_unique_knots_and_multiplicity` reported knots the space does not contain.
  The helper behind it rounded the knots onto the tolerance grid to decide which
  were the same and then returned the *rounded* values, although the indices it
  had built already pointed at the originals. On a float32 space over `[0, 1e-6]`
  with interior knots at 2.5e-7, 5.0e-7 and 7.5e-7 it answered 2.0e-7, 5.0e-7 and
  8.0e-7. `Bspline.multiply` builds the product's breakpoint mesh from this
  accessor, so squaring such a spline returned a curve wrong by 8% relative at
  points nowhere near a knot; `to_beziers`, knot insertion and removal and
  quasi-interpolation all read the same helper. The accessor now returns stored
  knots, and `_snap_knots` is implemented on top of it, so the space and its own
  accessor can no longer disagree about which knots are the same knot.
- `pantr.bspline.find_roots` returned values that are not roots, silently, on
  ordinary input: a clamped cubic on `[0, 1]` with control points alternating
  `+1, -1` gave back `0.375` and `0.625`, where the spline is `0.0208` rather
  than zero. Its residual test only ever ran on one of the four ways the
  tracking of a sign change can stop; on the other three the residual was
  hard-coded to zero and the value accepted unconditionally, and those are the
  branches that fire when the iteration is in trouble. Every exit now evaluates
  `|f(x)|` and the same test decides all of them, so a status records how the
  iteration stopped and never whether the value may be reported. Two further
  changes were needed to keep that from costing genuine roots. The stop on a
  repeated iterate now tests the hypothesis Mørken and Reimers actually state in
  their Corollary 14 — `degree - 1` active knots collapsed onto the iterate —
  rather than the bare repetition, which is also what a nearly horizontal
  control-polygon secant produces at a shallow non-zero minimum. And the
  threshold a tracked iterate is tested against now carries the term the
  parametric tolerance itself allows, `|f'| * 2 * tol * scale`, alongside the
  evaluation error: testing the evaluation error alone rejects the correctly
  located zeros of a steep spline. A side effect worth knowing: from degree 20 a
  double root came back as the two sign changes bracketing it, so a caller
  counting zeros got two where the answer is one. One of the pair was an
  uncertified budget-limited iterate and is now dropped, leaving the count right.
- The merge that collapses several reports of one zero could return a point
  between two distinct zeros. Reports are joined into a run on the *larger* of
  two merge radii, and the radius is capped at
  `domain_length * (degree! * zero_tol / coeff_scale) ** (1 / degree)`, which
  *grows* with degree — 0.114 at degree 9, 0.767 at 15, past the whole domain
  from 17 — so at high degree one radius joined every later root into its run.
  On a degree-15 spline with a C⁰ interior knot the three zeros found, each with
  a residual of 1e-17, came back as their midpoint alone, where the spline is
  `-0.64`. A merged midpoint is a value nobody tracked, so it now takes the same
  residual test as the reports it replaces, and a run whose midpoint fails is
  left as the separate roots it was. The radius policy itself is unchanged.
- Knot comparisons went through `np.isclose(a, b, atol=tol)` at 26 sites across
  seven modules. Setting `atol` does not clear `rtol`, which stays at NumPy's
  default `1e-5`, so the effective test was `|a - b| <= tol + 1e-5 * |b|`: on a
  domain of length 1 placed at `1e6` the tolerance was 10, not the `1e-15` the
  space reports. The consequences were a periodic space reporting the wrong
  number of basis functions, the in-domain gate admitting points ten domain
  lengths outside, `remove_knots` removing a different knot than the one asked
  for, `split` and `restrict` returning a piece that no longer interpolated its
  own end control point, and the exact extraction operator being discarded in
  favour of the identity on a knot vector that was near-uniform but not uniform.
  On a *translated* periodic domain it was also memory-unsafe: degree elevation
  indexed out of bounds, unchecked under `nopython`. All 26 now compare
  absolutely.
- Degree operations had no upper bound, while the exact-integer binomial
  recurrence they rely on overflows `int64` at `n = 62` — silently, since
  `nopython` does not trap it. `Bspline.elevate_degree`, `Bezier.elevate_degree`
  and Bézier composition returned corrupted control points from that degree on,
  where degree elevation must leave the curve pointwise unchanged. Two paths
  nobody would associate with degree elevation reach the same recurrence and are
  now capped too: `Bspline.derivative` on a rational spline, and
  `Bezier.minimize_degree`.
- Evaluating a 1D `Bspline` at points shaped `(n, 1)` — the shape the
  `(n_pts, dim)` convention implies, and the one this project's own docstring
  example uses — raised a Numba typing error from inside the kernel instead of
  being accepted. Points are now normalised before they reach any kernel, on all
  four entry points, and array-likes are coerced rather than failing on a
  missing `dtype` attribute.
- `SpanwiseElementExtraction.apply` and `apply_transpose` accepted an `out=`
  array aliasing their input and returned a silently wrong result; the alias
  check existed only for the two bilateral op kinds. `idx_map` was checked
  against its upper bound only, and only for non-identity entries, so a negative
  index reached the kernel and read out of bounds.
- `get_cardinal_intervals` on a degree-0 space read past the end of an empty
  array. `BVH` accepted node arrays deeper than the fixed traversal stack its own
  query kernels assume, and node arrays whose leaf markers and child pointers
  disagreed; both are rejected at construction now.
- `Bspline.reduce_degree` wrote past the end of its output buffers whenever the
  reduced Bézier form needed more control points than `len(knots)` allowed —
  from 13 elements on at degree 4, 8 at degree 5, 5 at degree 8, on the open,
  periodic and tensor-product paths alike. Unchecked under `nopython`, it
  surfaced as a control-point/basis-count mismatch from the `Bspline`
  constructor.
- The Bernstein evaluation recurrence seeded from `(1 - u)^p`, which underflows
  to zero for `u` close enough to 1 from degree 21 in float64 (degree 7 in
  float32); every later term is a positive multiple of the seed, so the basis
  summed to 0 instead of 1. The recurrence now starts from whichever endpoint
  keeps the seed above `0.5^p`. Degrees below the proven-safe bound keep the
  original branch-free loop, so the benchmark-gated tabulation is unaffected.
- The Cox-de Boor denominator guard compared a knot difference against an
  absolute tolerance, which is scale-dependent: on a small enough domain a
  genuinely non-empty knot span fell below it and was zeroed, breaking the
  partition of unity. The guard now tests against exact zero, which is
  scale-invariant by construction. What makes the exact test safe is the shape of
  the recurrence, not the knot vector: the denominator is the sum of the two
  non-negative terms the step then multiplies the quotient by, so the two weights
  are a convex combination and a tiny denominator cancels against an equally tiny
  numerator. (An earlier version of this entry justified it by claiming that knot
  snapping stores equal knots bitwise identical. Snapping does that, but the guard
  never needed it: with snapping off and two interior knots separated by 0 to
  64 ulp, `max|N|` is 1.0 exactly and the partition of unity holds to 6.7e-16.)
- `find_roots` reported a root at every interior knot of multiplicity
  `degree + 1` whose two straddling coefficients change sign, where the spline
  is C^-1 and jumps across the axis without ever reaching it. The tracking cited
  Mørken-Reimers Lemma 3, whose conclusion `c[a] = 0` rests on the iterate lying
  in the half-open interval between two consecutive Greville abscissae, and that
  interval is empty at precisely that multiplicity: the secant through the two
  coefficients is vertical there, so its zero is the knot for every `lambda` and
  nothing forces the coefficient to vanish. The fabricated root then took a real
  one with it, since reporting it split the spline at the jump and pinned the
  coefficient on the far side to zero as the split barrier, destroying the sign
  change that bracketed the next zero. On a quadratic split once at C^-1 the
  reported set was `[0.40269975, 0.5]` where the zeros are `0.40269975` and
  `0.61721778`. The lemma's hypothesis is now tested on the knot vector, which
  needs no tolerance, and a jump is rejected by the same residual test that
  already separates a tangential zero from a false sign change.
- `get_tanh_sinh_1d` returned `inf` for `1/sqrt(x)` — the integrand a
  double-exponential rule exists for — at every `n_pts >= 45`, and raising the
  point count therefore turned a correct answer into `inf`. A node whose
  distance to the endpoint had underflowed was moved *onto* the endpoint and
  kept, with a nonzero weight, and from `n_pts = 53` a second node landed on the
  same boundary and was returned twice. The rule is now truncated there instead:
  generation stops at the last node whose gap survives the mapping onto `[0, 1]`
  in the requested dtype, which is one machine epsilon. The discarded weight is
  at most `pi * cosh(t) * gap`, measured at `8.2e-15` against a weight sum of
  `2`, so a smooth integrand is unchanged; a singular one now converges to the
  truncation floor, `2e-8` for `x**-0.5` and `4e-15` for `log(x)`, both stated
  in the docstring. Returning fewer nodes than requested was already documented.
  In `float32` the cast onto `[0, 1]` collapsed a node onto `1.0` from
  `n_pts = 19`, which the dtype-aware threshold also closes.
- `QuadratureRule` claimed its factory-built rules integrate the constant `1`
  exactly. They do not, and cannot: dividing by a computed sum leaves the
  rescaled weights summing to `1` only up to rounding. Measured over Gauss-
  Legendre rules of 1 to 40 points per direction, `|sum - 1|` reaches 2 ulp in
  1D, 3 ulp in 2D and 4 ulp in 3D. The docstring now says that.
- Lagrange tabulation was not reproducible between processes. SciPy's
  `BarycentricInterpolator` permutes the nodes before forming the barycentric
  weights, which is Berrut and Trefethen's remedy against the product over- or
  underflowing, and it was built without the `rng` argument, so the permutation
  came from the unseeded global NumPy state. Two runs of the same call differed
  by 1.6e-16 to 1.5e-15 relative at degrees 3 to 12, by 4.18 absolute on a value
  scale of 3.75e7 at degree 62, and by `inf` against 1e16 evaluated outside
  `[0, 1]`. That is an unseeded generator, not floating-point nondeterminism
  with a bound. `tabulate_lagrange`, `compute_lagrange_to_bernstein_1d`,
  `tabulate_Lagrange_extraction_operators` and `SpanwiseElementExtraction` with
  a Lagrange target all inherited it. The permutation is now seeded from a
  recorded constant, which also makes the `degree + 1` cardinal functions share
  one set of weights instead of drawing their own.
- `Bspline.elevate_degree` and `Bspline.reduce_degree` were unusable on every
  float32 spline. Both kernels allocated the two halves of their return with
  different dtypes: the control points followed the input while the knot vector
  was hardcoded float64. On a clamped space the `Bspline` constructor then
  rejected the mismatched pair with a message about the *caller's* control
  points, at every degree and every knot count; on a periodic space the round
  trip through open form converted the control points as well, so the call
  succeeded and silently discarded the caller's choice of precision, doubling
  memory and halving throughput without a word. Both knot outputs now follow the
  input knot vector's dtype.
- `Bspline.reduce_degree` never returned on a degree-1 periodic spline. Degree
  reduction hands the periodic conversion `m_bdy - decrement` as the target
  boundary multiplicity, and a maximally smooth periodic space has `m_bdy = 1`,
  so the argument is 0 at every degree; the conversion documents
  `1 <= m_bdy <= degree` and nothing checked it. The ghost-knot builder then had
  a per-period tile of total multiplicity 0 and its right-hand loop, which
  appends until it has enough ghost knots, could append nothing and incremented
  its shift forever. Degrees 2 and 3 escaped only because the C^0 seam check a
  few lines earlier rejected them first, so the trigger's narrowness was an
  accident of check ordering rather than anything about degree 1. The
  precondition is now enforced, so every degree refuses in under 10 ms with a
  message naming the actual problem; the loop's termination argument is written
  down beside it. Reducing a periodic spline to degree 0 has no answer in this
  representation, an empty `[1, degree]` meaning no ghost knots and nothing to
  wrap; `to_open_bspline().reduce_degree(1)` does it in the open form.
- `Bspline.multiply` built the product space from `max(m_f + q, m_g + p)` at a
  breakpoint both operands share and from a hardcoded `p + q` at one only a
  single operand carries. The second form assumes the present operand has
  multiplicity `p`, a hypothesis the module never stated, and it is wrong in both
  directions. Against a discontinuous operand (multiplicity `degree + 1`, which
  `subdivide(n, regularity=-1)` produces) the knot vector came out one knot
  short, the intermediate Bézier space did not contain the product, and the
  underdetermined Oslo system returned a least-squares vector unrelated to it:
  multiplying such a spline by the constant 1 gave a different function.
  Exhaustive over `p, q <= 3` with multiplicities up to `degree + 1`, 45 of 81
  configurations were wrong, every one of them with a discontinuous operand.
  Below multiplicity `p` the same line asked for up to `p - m_f` knots more than
  the product needs, so the ordinary continuous case now yields a smaller space
  as well: degree 4 times degree 1 across a C^3 knot builds 8 control points
  where it built 11. The rule that replaces both, minimal by Curry-Schoenberg, is
  `(p + q) - min(s_f, s_g)` with `s = degree - m` the smoothness of an operand
  there and `s = +inf` where it has no knot at all. It now lives in one function
  that the 1D and tensor-product paths both call.
- `Bspline.elevate_degree` returned control points and a knot vector that
  disagreed on how many basis functions the elevated space has, by one per
  interior knot of multiplicity `degree + 1`. Piegl-Tiller A5.9 starts each
  segment at its second elevated coefficient because the segment before it
  already wrote the shared junction value; where the spline is discontinuous the
  two segments share nothing and that coefficient was dropped. At degree 0 a
  second cause compounded it: the segment sweep relied on the closing block of
  `degree + 1` equal knots to reach the end of the knot vector, which at degree 0
  is a single knot, so the last segment and the closing knots were never written
  and the elevated spline came back on a domain collapsed to a point, without an
  error.
- `Bspline.derivative(direction, keep_degree=True)` raised
  `ValueError: The number of control points must be a multiple of the number of
  basis functions` for every spline with a C⁰ interior knot, which is the
  ordinary Bézier-element mesh rather than an exotic input. It differentiates to
  `degree - 1` and elevates back, so a knot of multiplicity `degree` becomes one
  of multiplicity `(degree - 1) + 1` in the space handed to the elevation kernel:
  the discontinuous case fixed above. It now returns the derivative, which is
  itself discontinuous there, carrying multiplicity `degree + 1`.
- `Bspline.reduce_degree` fitted a discontinuous knot into a C^0 space. Reduction
  preserves smoothness, so multiplicity `m` becomes `m - t`; the target was
  clamped at `new_degree`, one short of the `new_degree + 1` a jump needs, and
  the kernel underneath merged the two sides of every junction into one control
  point on the strength of an endpoint-interpolation argument that holds only
  where the original spline is continuous. On input built to be exactly
  reducible, the reduced curve now reproduces it to 4.4e-16 or better at degrees
  2 to 4, where the errors were 5.7e-01, 1.0e+00 and 4.9e-01.
- `Bspline.reduce_degree` on a maximally smooth periodic spline asked the
  periodic conversion for a seam multiplicity of `m_bdy - decrement`, which is 0,
  while interior breakpoints three lines away were already floored at 1. It now
  uses the same floor, which asks for less smoothness than the seam already has
  and is therefore always representable, and a maximally smooth periodic
  quadratic reduces to degree 1. From degree 3 the call still fails, because the
  segment-wise reduction does not preserve C^1 across the seam, but on the
  residual test that says so rather than on the knot vector. The degree-0
  rejection above is unaffected and now fires only where its reasoning applies.

### Documentation
- **Every docstring example in `src/pantr` runs, and CI runs them.** `make doctest` is a
  new target (`pytest --doctest-modules src/pantr`, with `NUMBA_DISABLE_JIT=1`), wired
  into `make pre-pull-request` and into the CI test job on the full Python matrix, since
  an example can drift from the code on any version and a numpy release that prints a
  value differently breaks it on one leg only. Disabling the JIT keeps the run clear of
  the warmup thread the package starts at import, and takes 0.9 s against 24 s cold.
  Twenty-one Example blocks across `Bezier` and `Bspline` were pseudo-code over names
  that never existed, so every one of them raised `NameError` the moment it was run;
  each now builds its own geometry in a line or two and asserts a characterizing fact.
  No example matches a printed numpy array any more: values go through `np.allclose`
  against a short literal and index arrays through `.tolist()`, the convention
  `change_basis` already used, so a formatting change in numpy cannot break the
  documentation. Three examples were showing wrong *values* rather than wrong
  formatting: all three of `BsplineSpace1D.get_cardinal_intervals` (doctest reports only
  the first failure, which masked the other two) and one in the cardinal tabulation
  kernel, which also called itself by a name that does not exist in its module. The code
  was right in each case, and is untouched; `get_cardinal_intervals` now states the rule
  it implements and why its window is the right one.

## 0.6.0 (2026-06-24)

### Added
- `pantr.mpi.configure_threads`: explicitly set the per-rank Numba thread count
  for hybrid MPI + threads runs (optionally also limiting BLAS/LAPACK thread
  pools via `threadpoolctl`). Calling it disables the default MPI
  thread policy.

### Changed
- MPI-engaging entry points (`pantr.mpi.DistributedSpace`,
  `pantr.mpi.from_dolfinx`) now apply a process-level default on first use: the
  Numba thread pool is limited to one thread per rank (flat MPI), preventing
  `n_ranks x n_cores` thread oversubscription when running under `mpiexec`.
  Explicit configuration always wins (`NUMBA_NUM_THREADS`,
  `pantr.set_num_threads`, `pantr.num_threads`, or
  `pantr.mpi.configure_threads`), and the policy is applied at most once per
  process, so raising the count afterwards sticks.

### Removed
- Dropped the `THIRD_PARTY_NOTICES` file and its `pyproject.toml` `license-files`
  entry. The routines that previously followed algoim — Bézier/B-spline degree
  reduction, Bernstein interpolation, the Bernstein L2 norm and degree
  minimization, and the tanh–sinh rule with its Lambert W step — were
  reimplemented clean-room from public references (The NURBS Book; Farouki &
  Rajan, *CAGD* 1988; Golub & Van Loan; Takahasi & Mori 1974; SciPy), with no
  change in public API or numerical results, so the third-party attribution is
  no longer required.

## 0.5.1 (2026-06-03)

### Added
- `pantr.grid.overlay`: build the coarsest `TensorProductGrid` that refines two
  input tensor-product grids — its per-axis breakpoints are the union of both
  inputs' breakpoints restricted to their domain overlap, so every overlay cell
  lies inside one cell of each input. Defined for any `ndim >= 1`. The
  background-grid bridge for immersed / unfitted quadrature.

## 0.5.0 (2026-06-03)

### Added
- `pantr.grid`: new structured-grid layer. `Grid` is an abstract base class
  defining the grid contract (cell bounds, point location, facet neighbours)
  with axis-aligned box defaults for facets, reference maps, neighbour lists,
  batch point location, and AABB queries. `TensorProductGrid` is a concrete,
  low-footprint tensor-product grid of axis-aligned boxes with per-axis
  breakpoints and row-major (C-order) cell ids matching
  `SpanwiseElementExtraction`. Factories `uniform_grid` and `tensor_product_grid`
  build a uniform grid and a B-spline knot-span grid respectively. `BVH` is a
  bounding-volume hierarchy over cell AABBs (lazily built, backing
  `Grid.query_aabb`), and `CellTags` / `FacetTags` are sparse, dolfinx-style
  named tag registries for cells and facets.
- `pantr.grid.HierarchicalGrid`: hierarchical refinement grid with a fixed
  per-direction subdivision factor (octree = the dyadic case). Active cells are
  stored as rectangular blocks per level (no per-cell storage); supports
  `refine(level, lo, hi)` with union semantics, automatic single-level balance,
  `refine_cells`, and `hanging_neighbors` for hanging-node facets. Built with the
  `hierarchical_grid(root, factor)` factory.
- `pantr.viz.grid_to_pyvista`: export a 1-D/2-D/3-D `Grid` to a pyvista
  `UnstructuredGrid` (lines / quads / hexahedra).
- `pantr.quad.QuadratureRule`: immutable d-dimensional quadrature rule on the
  unit cube `[0, 1]^ndim`, with `tensor_product_quadrature` (tensor product of
  per-axis 1-D rules) and `gauss_legendre_quadrature` (isotropic or anisotropic
  Gauss-Legendre) factories.
- `pantr.grid.cell_quadrature`: map a reference `QuadratureRule` from the unit
  cube onto a grid's cells (or a subset), returning per-cell points
  `(num_cells, num_points, ndim)` and weights `(num_cells, num_points)` via the
  per-cell affine map with volume-scaled weights. The uncut/background-cell
  quadrature bridge for immersed / unfitted discretizations.

## 0.4.0 (2026-06-02)

### Added
- `pantr.geometry`: new module exposing `AABB`, an immutable, general-*d*
  axis-aligned bounding box (#153). Shared domain primitive for spline-space
  parametric domains and grid-cell bounds; decoupled from any concrete affine
  transform via a structural `_AffineMap` protocol.

### Changed
- `pantr.transform.AffineTransform`: stricter input validation (reject
  zero / non-finite scaling factors; validate rotation-axis and mirror-normal
  finiteness and the `center` shape), a cached `inverse`, and C-contiguous
  stored arrays (#154). Enables a downstream consumer to adopt pantr's `AffineTransform` and
  drop its local copy.

## 0.3.0 (2026-05-06)

### Added
- `SpanwiseElementExtraction` class providing a unified interface for element-wise
  extraction operators across B-spline spaces (#143).
- Batch apply methods on `SpanwiseElementExtraction` for vectorized evaluation (#145).
- Numba-callable Kronecker kernels backing tensor-product extraction (#140).
- Structural identity predicate for Bezier and Lagrange extraction operators (#147).
- Numba-callable struct-view of `SpanwiseElementExtraction` for downstream JIT code (#149).
- Python 3.13 and 3.14 are now officially supported (#151).
- User guide for `SpanwiseElementExtraction` (#146).

### Changed
- `nD` Bezier extraction is now routed through `SpanwiseElementExtraction`,
  unifying the 1D and multi-dimensional code paths (#144).

### Performance
- Compact storage for identity-heavy extraction spaces (#148).
- CI pipeline and test suite sped up (#150).

## 0.2.0 (2026-04-19)

### Added
- `pantr.bezier`: mask / boolean-array operations (#112), Sylvester and Bezout matrix
  construction (#113), determinant and rank via Givens-rotation QR (#115),
  `Bezier.interpolate` / `Bezier.fit` classmethods (#118, #120),
  resultant / discriminant / `minimize_degree` (#121), pure-Numba implicit quadrature
  module (#128, #130, #131), implicit domain reparameterization with Lagrange cells
  (#132).
- `pantr.bspline`: interpolation, fitting, and L2 projection (#122, #124).
- `pantr.quad`: modified Chebyshev nodes for Bernstein interpolation (#114),
  tanh–sinh quadrature rule (#116).
- `pantr.root_finding`: Bernstein polynomial root-finding module, with a unified
  single / batch API (#111, #123). First contribution by @DavorDobrota.

### Changed
- Public function renames in `bspline`, `bezier`, and `cad` for clarity (#125, #126).
- Conversion classmethods extracted as standalone module functions in `bspline`
  and `bezier` (#127).
- `change_basis`: added `compute_monomial_to_bernstein_1d`, reused across
  `bezier` and `bezier.implicit` (#135).
- `bezier`: `_gauss_legendre_01` now delegates to `pantr.quad` (#133).
- Layer 2 validation helpers consolidated and shared across `bezier` / `bspline`
  (#138, #139).
- `bezier.implicit`: legacy algoim engine moved out to a downstream consumer,
  and dead algoim-era modules dropped (#136, #137).

### Documentation
- Added algoim attribution and third-party notice for the implicit quadrature
  module (#134).

## 0.1.0 (2026-03-24)

- Initial release: project scaffolding, tooling configuration, and
  documentation skeleton, plus the core `basis`, `bspline`, `bezier`, `quad`,
  `change_basis`, `cad`, and `viz` modules.
