# The root-finding seam

Decision record for how pantr finds the roots of a Bernstein polynomial and how another root
finder (BeRF-FR, through the `pantr-berf` package) plugs in. Expensive to reverse because the
claim and completeness codes are a wire format between extension modules, and every caller of
`find_roots` reads its warnings from them. Architecture: `design/port_design.md` §2.7 and §4.1;
code: `include/pantr/roots/`.

## What a result certifies

The object is named in the input's terms: the roots in `[0, 1]` of the stored scalar Bernstein
polynomial; for a rational Bézier, of its first homogeneous component (whose roots are the
rational function's where the weights are positive, the library's convention).

- Each root carries a `Claim`: `none = 0`, `computed = 1` (returned by the algorithm, nothing
  more), `certified = 2` (existence and uniqueness of a root within `param_tol` of `x` proved).
  A result is only as certified as the weakest step that produced it.
- Each result carries a `Completeness`: `complete = 0`, `gap_localized = 1` (the listed intervals
  carry no claim), `gap_unlocalized = 2` (some part of `[0, 1]` carries no claim, place unknown).
- The integer values are pinned: they cross into Python and between extension modules, and a
  `static_assert` and a Python test hold them.
- `Capabilities` is a value (`max_claim`, `batch`, `monotone`, `multiplicity`). No minimum claim
  can be requested: a request a provider cannot honour would need a failure mode of its own, and
  no caller needs one yet (on the later list).
- Frames: `param_tol` is absolute on the parameter interval `[0, 1]` (the bracket width at which a
  root counts as found); `value_tol` is absolute in coefficient units (the floor of every zero
  test). The Python default passes the dimensionless `get_strict(dtype)` into both, as the
  previous release did; whether it should scale with the coefficients is left to the end review.
- Roots are `double` whatever the coefficient type: `float32` polynomials have always returned
  `float64` roots. Design §4.1 wrote `RootSet<T>`; the parameter type is `double` instead.

## The Builtin provider

pantr's own provider is the previous release's algorithms, ported operation by operation:
Yuksel's monotone decomposition below degree 6 or past a coefficient range of `1e8`, Bezier
clipping otherwise, and the derivative-aware merge of clipping's candidates. Its roots are the
previous release's bit for bit on both dtypes (work arrays keep the storage type, and no product
is contracted into an `fma`, since the branches are sign tests a last bit can flip), except where
the previous release's candidate buffer overflowed and dropped roots, and in the depth-cap case.

- Every root is `computed`; no multiplicity is claimed (`-1`).
- Completeness is `complete` unless clipping gave up on an interval at its depth cap (64): that
  interval is then listed and the result is `gap_localized`. The Python wrapper turns it into a
  `RuntimeWarning` (QB7). A polynomial whose clips are exact (a line written at a higher degree)
  with a tolerance below the spacing of doubles at its root reaches the cap, since no interval can
  then narrow to it. `gap_unlocalized` is never produced.
- The Newton/bisection hybrid's iteration cap (64) is not a completeness gap: it keeps a bracket
  that holds the root and returns its midpoint, so only width is lost, never the root.
- The Numba kernel's fixed buffers are gone: a work stack of 256 (it could not bind, a depth-first
  stack holding at most `depth + 2` entries) and a candidate buffer of `3n + 4`, which did bind
  (polynomials with many exact, evenly spaced roots filled it, and roots were lost) and now grows.
- The single and batch dispatches keep the previous release's two differences: the batch merges
  Yuksel's roots too, and its all-zero test passes a NaN coefficient.

## Duplicate pair classes

Merging duplicates must not merge distinct roots. Each class of duplicate, the layer that removes
it, and the test that asserts one root per true root (`cpp/tests/test_roots.cpp`):

| Pair class | Produced by | Removed by |
|---|---|---|
| A root at an endpoint shared by two adjacent subintervals | clipping's endpoint check (step 6) on both intervals, the same value twice | the merge's base radius `max(2 param_tol, 4 zero_tol)` (gap 0) |
| A multiple root reached by two clip branches | two converged intervals within `O(zero_tol^(1/m))` of a root of multiplicity `m` | the derivative-aware radius `min(4 zero_tol / max(|f'|, eps), zero_tol^(1/3))`, aimed at `m <= 3`; it misses a triple root whose other factors are small there (see below) |
| A root at 0 or 1 found as an endpoint and by convergence | the boundary check before clipping, and the polished midpoint of the last interval | the base radius (the polish clamps to `[0, 1]`) |
| A root at a critical point of Yuksel's walk | the endpoint test of two adjacent monotone intervals | the walk's own guard (an endpoint within `tol` of the last root is not added), in some placements only; the batch merge removes the rest |

Two genuinely distinct roots closer than the merge radius are reported as one; the radius is the
price of reporting each double root once.

**What the table does not yet hold**, all inherited from the previous release and pinned by
expected-failure tests in `cpp/tests/test_roots.cpp` (`doctest::should_fail`, to be removed with
the repair):

- Yuksel's single-polynomial path does not merge, so a double or triple root below degree 6 can
  be reported more than once (a triple root at `1/2` beside roots at `1/8` and `7/8` gave five
  roots for three). The batch path merges and reports each once.
- A triple root on the clipping path whose cofactor is small there, for example
  `(t - 2/5)^3 (t - 1/10)(t - 9/10)(t + 1)`, is reported twice: its candidates sit at both edges
  of the band `|f| <= zero_tol`, further apart than the cube-root cap.
- The base radius `max(2 param_tol, 4 zero_tol)` compares a parameter gap with a coefficient-unit
  quantity, so at a large coefficient scale (from about `1e12` in `double` at the default
  tolerance) distinct roots merge, the result reading `complete`. This is the frame question the
  end review's tolerance pass owns; the remedy is a radius in parameter units.

## A provider in Python

`pantr.bezier.set_root_provider("builtin" | "berf")` and `get_root_provider()` select the provider
for the process; the initial value comes from `PANTR_ROOT_PROVIDER` on first use (an unknown name
raises `ValueError` there, not at import), else `"berf"` when `pantr_berf` imports, else
`"builtin"`. A provider module exposes `find_roots`, `find_roots_batch`, `find_monotone_root` and
`find_monotone_root_batch`, each with an `_f32` twin, taking C-contiguous coefficient arrays and
returning `(roots, completeness)`, `(roots, counts, completeness)`, a float, and an array. The
BeRF-FR adapter (stage S3) maps BeRF-FR's per-row certificate through one exhaustive `switch`
with no default, so a new BeRF-FR enum member fails its compile.

## Alternatives not taken

- Keeping the Numba caps as silent truncation: a result that dropped work would read as complete.
- A requested minimum claim in the options: no caller needs it, and it needs a refusal mode.
- `float` roots for `float` coefficients (§4.1's letter): fewer digits than the API returned.
