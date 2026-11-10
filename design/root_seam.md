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
  root counts as found), and it is the user's `tol`; `value_tol` is relative to each polynomial's
  largest coefficient magnitude (the floor of every zero test is `value_tol * max|c|`, per row of
  a batch), and the Python layer always passes `get_strict(dtype)`. A provider that read
  `value_tol` in coefficient units would make its roots depend on the scale of the coefficients.
- Roots are `double` whatever the coefficient type: `float32` polynomials have always returned
  `float64` roots. Design §4.1 wrote `RootSet<T>`; the parameter type is `double` instead.

## The Builtin provider

pantr's own provider is the previous release's algorithms, ported operation by operation except
for the monotone solver (`find_monotone_root`), whose safeguarded solve is new (below):
Yuksel's monotone decomposition below degree 6 or past a coefficient range of `1e8`, Bezier
clipping otherwise, and the derivative-aware merge of clipping's candidates. Its roots are the
previous release's on both dtypes where the previous release's coefficients were of order one
(work arrays keep the storage type, and no product is contracted into an `fma`, since the branches
are sign tests a last bit can flip). It departs from them where the previous release's candidate
buffer overflowed and dropped roots, in the depth-cap case, and wherever its absolute value-frame
bounds mattered: the zero tests, the flatness test, the Newton and multiple-root derivative tests
and the merge radii are all multiples of `max|c|`, so `c` and `2^k c` give the same bits wherever
nothing under- or overflows.

- Every entry (single, batch, monotone, monotone batch) refuses a NaN or an infinite coefficient
  (`value_error`, the first such row named in a batch) and a `param_tol` or `value_tol` that is not
  positive and finite (NaN included), before anything is computed: for non-finite coefficients no
  root, and no "no root", is an answer, and the previous release gave either. The dispatch beneath
  the entries therefore sees finite coefficients only, and carries no NaN handling of its own.
- Every root is `computed`; no multiplicity is claimed (`-1`).
- Completeness is `complete` unless clipping gave up on an interval at its depth cap (64), or a
  Newton/bisection solve of Yuksel's walk reached its iteration cap (below): that interval (the
  solve's final bracket) is then listed and the result is `gap_localized`. The Python wrapper turns it into a
  `RuntimeWarning` (QB7). Reaching it needs a polynomial whose clips are exact (some lines
  written at a higher degree, at some roots), a tolerance below the spacing of doubles at its
  root, and a value floor small enough that no interval reads as flat first: at the floor the
  Python layer passes, no input is known to reach it. `gap_unlocalized` is never produced.
- The two Newton/bisection hybrids have an iteration cap, and reaching it is a completeness gap.
  The monotone solver (`find_monotone_root`) bisects where Newton fails: when the step leaves the
  bracket, when the derivative is zero or not finite, and when the iteration just made, itself not
  a bisection, failed to halve the bracket. The bracket then at least halves every two iterations,
  so its cap, `max_safeguarded_iter` = `2 (ceil(log2(1 / tol)) + 1)` at the strict `float64`
  tolerance `2^-50`, that is 102 for both dtypes (the bracket is a pair of doubles), reaches any
  `param_tol` from `2^-50` up whatever Newton does. It is reached only for a finer `param_tol`, at
  the latest one below the spacing of doubles at the root; the bracket's midpoint can then be far from the root, so the solver raises
  (`state_error`) instead of returning it. The walk's hybrid keeps the previous release's cap of
  64, an admitted bound: it has no halving guarantee.
- `find_roots`' interval walk (Yuksel's, per monotone interval) keeps the previous release's hybrid
  without the bisection safeguard: its guard against reporting a root twice at a critical point
  relies on where that hybrid ends near a multiple root, and the safeguarded steps end elsewhere
  within the root's rounding band, so a triple root is then reported twice (see
  `solve_on_interval`). A one-sided stall there (a `float32` evaluation near a simple root, a
  multiple root) reaches the cap; the walk lists the final bracket as a gap, so `find_roots` can
  warn on a `float32` simple root that `find_monotone_root` solves.
- `find_monotone_root` of end values without a sign change returns NaN, "no root" as a value (a
  batch row, NaN too); non-finite coefficients never reach it.
- The Numba kernel's fixed buffers are gone: a work stack of 256 (it could not bind, a depth-first
  stack holding at most `depth + 2` entries) and a candidate buffer of `3n + 4`, which did bind
  (polynomials with many exact, evenly spaced roots filled it, and roots were lost) and now grows.
- The single and batch dispatches keep the previous release's one difference: the batch merges
  Yuksel's roots too. The previous release's other, a batch zero-polynomial test that passed a NaN
  coefficient, went with the refusal of non-finite coefficients, as did the single path's NaN
  ordering of Yuksel's roots and its NaN range flag. The zero-polynomial test is exact (every
  coefficient zero): with a relative floor, nothing else is within it.

## Duplicate pair classes

Merging duplicates must not merge distinct roots. Each class of duplicate, the layer that removes
it, and the test that asserts one root per true root (`cpp/tests/test_roots.cpp`):

| Pair class | Produced by | Removed by |
|---|---|---|
| A root at an endpoint shared by two adjacent subintervals | clipping's endpoint check (step 6) on both intervals, the same value twice | the merge's base radius `max(2 param_tol, 4 z)` (gap 0), `z = zero_tol / max|c|` |
| A multiple root reached by two clip branches | two converged intervals within `O(z^(1/m))` of a root of multiplicity `m` | the derivative-aware radius `min(4 zero_tol / max(|f'|, eps max|c|), z^(1/3))`, aimed at `m <= 3` |
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

The two clipping-path entries this list held, a triple root with a small cofactor reported twice
and distinct roots merged at a large coefficient scale, were both the merge radii's coefficient
units; with the radii relative to `max|c|` their tests are ordinary ones.

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
