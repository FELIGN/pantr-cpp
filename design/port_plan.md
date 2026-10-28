# Plan: the pantr C++ port

The approved plan for porting pantr from Python/Numba to a header-only C++20 library with a thin
nanobind binding, built on the branch `feat/cpp-port` and merged as one pull request with one commit
group per stage. The architecture is `design/port_design.md` (cited here as §n). Identifiers used
below:

- **R0-R11** are the maintainer's rulings, listed under "Rulings" below.
- **AC1-AC13** are the acceptance criteria of the whole port, listed under "Acceptance criteria".
- **DR-n** are this plan's decision records (end of file); **Di1-Di6** are directives inferred while
  planning and approved with the plan (R11); **QB1-QB11** are defaults the plan proceeds on unless
  overruled; **P0a, P0b, P1, ...** are the build items; **S0-S9** the stages.

## Rulings

- **R0** The port's fixed decisions: one backend (C++ replaces Python/Numba module by module, no
  dual backend, no parity harness; the existing Python suite is the oracle, with derived tolerances
  where floating-point results move, and mirror tests replaced by independent oracles, never
  deleted); scope is everything including `pantr.mpi`, with `pantr.viz` staying Python; the public
  Python API is unchanged except where C++ forces a change, each such change justified and listed
  in `docs/changelog.md`; C++ is a library in its own right (stable headers, `find_package(pantr)`,
  usable with no interpreter); pantr stays MIT, with root finding behind a seam whose
  higher-capability provider (BeRF-FR) ships only in a separate `pantr-berf` wheel; no dependency
  that would stop pantr being MIT; `pantr::pantr` and `pantr::mpi` are header-only INTERFACE
  targets; contraction (`-ffp-contract=on`) and `-march=native` propagated by the INTERFACE target,
  both optional (the latter decided in the consumer, DR-9); SIMD chosen by macros with a scalar fallback; multi-ISA wheels; dependencies
  C++20, Threads, Eigen, Spectra, Kokkos mdspan where `<mdspan>` is missing; pantr's own
  `std::thread` pool sized from the affinity mask (no OpenMP, no TBB); METIS optional behind a seam;
  numpy and scipy (scipy only at the Python boundary) at runtime, Numba and threadpoolctl removed;
  Python >= 3.12; nanobind and scikit-build-core; distributed logic in C++ (`pantr::mpi`) with core
  headers never including `mpi.h`, threads and MPI combined as `MPI_THREAD_FUNNELED`, `_pantr_mpi`
  a separate extension following dolfinx's wheel scheme (one MPI family per platform, MPI kept out of
  the wheel, an import-time family check).
- **R1** A grid written by the user is no longer possible: `Grid` cannot be subclassed from Python.
- **R2** Fitting scope is exactly what exists today; extensions go to the "Later" list.
- **R3** No automatic-differentiation type; kernels are generic over the scalar.
- **R4** Wheels: `NB_STATIC` + stable ABI (cp312 abi3), `NOMINSIZE`, `NB_SUPPRESS_WARNINGS`,
  `install.strip = false`, nanobind pinned exactly across all extension modules.
- **R5** C++ domain types are immutable; the Python API is unchanged by swapping the held C++ value.
- **R6** Keep `NB_STATIC` + abi3 and measure the per-call overhead early; split mode only if a tiny
  call breaks AC6.
- **R7** BeRF-FR is fetched from its public repository at a pinned SHA; its stage runs late, and
  returns to the maintainer if the repository is still private then.
- **R8** Platforms: Linux x86_64/aarch64, macOS arm64, Windows amd64 for `pantr` and `pantr-berf`;
  `pantr-mpi` on Linux and macOS.
- **R9** Constructors and `copy=True` conversions are exempt from AC6 for their O(n) copy term.
- **R10** The repository is made public before the branch's first CI push.
- **R11** The plan below was approved as written, including Di1-Di6 and the QB defaults.

## Acceptance criteria

- **AC1** Python suite, doctests and real-MPI tests (`mpiexec -n 2` and `3`) pass on the branch's
  CI; every changed assertion is one of: derived tolerance, independent oracle replacing a mirror,
  replaced Numba-specific test, or an API change listed in the changelog.
- **AC2** The docs build with `-W` passes on CI.
- **AC3** The C++ test suite over the public headers passes on CI under GCC and Clang, plus
  ASan/UBSan and TSan legs; TSan covers the thread pool and every lazy derived cache.
- **AC4** A CMake consumer with no Python builds against an installed pantr via
  `find_package(pantr)`, linking `pantr::pantr` and `pantr::mpi`; core headers never include
  `mpi.h` (a CI job builds and runs the consumer; an include check fails on `mpi.h` in the core).
- **AC5** numba and threadpoolctl are gone from dependencies and source; scipy is imported only in
  binding-boundary modules (an import check in CI).
- **AC6** No public operation is slower than on `main` with Numba, on the benchmark list of §9.1:
  per operation, branch median <= main median + main's own measured spread, BLAS and pool threads
  pinned (statistic fixed in DR-3).
- **AC7** The `pantr` wheel and the default C++ build contain no BeRF-FR header; `pantr-berf` builds
  as a separate wheel; `pantr[berf]` enables it; a runtime switch disables it;
  `-DPANTR_WITH_BERF=OFF` by default.
- **AC8** Multi-ISA wheels: per-ISA extension, baseline cpuid probe, import-time ladder, environment
  override with no fallthrough (a test per rung reachable on the CI runner).
- **AC9** Every public Python API change is in `docs/changelog.md` with its justification.
- **AC10** `_pantr_mpi` is a separate extension; `import pantr` works with no MPI installed; an
  import-time check reports a missing or mismatched MPI.
- **AC11** METIS is optional: `-DPANTR_WITH_METIS=ON` and `pantr[metis]`; spectral partitioning
  otherwise.
- **AC12** Every new third-party dependency is named in this plan with its licence; none is
  incompatible with MIT.
- **AC13** Each stage ends green on full branch CI before the next starts; one pull request, one
  commit group per stage.

## How every stage runs
- One worktree for the branch, off `main` 11ab436, in the `pantr-cpp` conda environment; the
  extension is built in place with `make ext` (no editable install, §1.9).
- Stages are sequential (dependencies between them, and a cap on concurrent build agents). Each
  build item is built by one agent that runs its own review loop and commits on the branch.
- A stage is done when: its components are built; every Numba kernel whose last caller it ports is
  deleted (DR-2: kernels with unported callers survive, listed per stage); its tests pass
  with every changed assertion in one of the four AC1 categories (derived tolerance, independent
  oracle replacing a mirror, replaced Numba-specific test, changelog-listed API change), listed in
  the commit group; the **whole** benchmark list (§9.1) passes under the interleaved protocol of
  DR-3 (copy terms exempt, R9); the downstream-consumer census was run (below); its changelog
  entries are in; the branch is pushed and full CI is green. No stage starts on a red gate (AC13).
- Before every push: `conda run -n pantr-cpp make pre-pull-request` over the whole repo
  (`CLAUDE.md`), extended in P0b with the C++ build and ctest. The branch is pushed once per stage,
  at its end (S0 is pushed once, after P0b; P0a's first commit already carries the CI trigger and
  the repository is public before that push, DR-18).
- Before a stage deletes or reshapes any public or private symbol: the downstream-consumer census
  (public names, private names, attribute writes), recorded in the commit group; a hit returns to
  the maintainer (`CLAUDE.md`).
- Commits are conventional (`<type>(<scope>): <summary>`); a stage's review-loop and repair commits
  keep that form with "loop-review" or "repair" in the body.
- Check per item: the builder's own review loop, plus an independent read of its diff, its
  loop-review commits and its open suspicions, plus green full CI.

## Standing constraints every item carries
- C++20; header-only INTERFACE targets; kernels `template <class Real>` with unqualified math via
  `using`, domain types over float/double (R3, design §2.3); immutable values with shared const
  representation (R5, §2.2).
- nanobind modules `NB_STATIC STABLE_ABI NOMINSIZE NB_SUPPRESS_WARNINGS`, `install.strip = false`;
  the same exact nanobind pin for `_pantr`, `_pantr_mpi` and `_pantr_berf` (R4, §1.9).
- pantr.viz stays Python on the new public API (§3.7); viz tests run headless every stage.
- Never `pytest -n auto` here: Makefile `test` defaults to `-n 8`, CI passes `auto` explicitly.

## Order
S0 foundation → S1 quad/tolerance/basis/change_basis → S2 bezier + geometry/transform + root seam
→ S4 bspline core + spanwise extraction (DR-2) → S5 fitting + remaining extraction → S6 grid/locate/THB/partition → S7 cad/multipatch
→ S8 mpi → S3 BeRF-FR provider (late: R7) → S9 Numba retirement → end review.

## Items (components from §10)
- P0a · first commit: ci.yaml triggers on push of feat/cpp-port, Makefile `-n 8` default
  (DR-18). Then C0.1 CMake/options/probes/deps/install-export/consumer, C0.2 core headers
  (config, errors + translator, `py_repr`, ndarray, mdspan alias, lazy, simd, exact rational shim),
  C0.3 pool/parallel_for/affinity/work model + `pantr._parallel` on C++. Also commits this plan as
  design/port_plan.md and the design notes §13 (1), (6), (7), (8).
- P0b · C0.4 packaging (pyproject on scikit-build-core, nanobind pinned exactly (3 modules),
  NB_STATIC+STABLE_ABI cp312, multi-ISA, `_pantr_cpu`, `_isa.py`, `_ext.py` guards, make targets,
  stubs, Windows/MSVC path: R8), C0.5 CI rewrite (§8.2 legs 1-12 plus MSVC; berf leg 13 lands in
  S3; triggers on push of the branch, as P0a set up), C0.6 benchmark harness + main baseline capture (§9.3) +
  abi3 measurement (§9.4, R6). **Return to the maintainer if, on any `tiny` case, the abi3 build exceeds
  main's DR-3 statistic and exceeds the version-specific build by more than main's range (R6).**
  Design note §13 (5).
- P1 · planned first, then built in batches · C1.1 quad incl. Lambert W, C1.2 tolerance + basis
  incl. barycentric Lagrange (the barycentric formula in C++, no scipy), C1.3 change_basis, C1.4 bindings/wrappers/tests (M2, warmup and
  parallel test replacements §8.4). Design note §13 (2).
- P2a · C2.1 AABB/AffineTransform, C2.2 Bezier value type and operations with handle swap.
- P2b · C2.3 degree reduction (exact rational, Boost.Multiprecision) and interpolate/fit,
  C2.4 root seam + Builtin provider + `set_root_provider` (QB7 below), C2.5 tests (M3, M6, U3 check).
  Design note §13 (3).
- P4a · C4.1 spaces/knots/basis core/factories/Greville + their tests.
- P4b · C4.2 Bspline value type and operations (M1, M5) + their tests, plus C5.2's
  SpanwiseElementExtraction and the one contraction primitive (to_beziers needs it, DR-2).
- P4c · C4.3 B-spline roots (status → RuntimeWarning) and products + remaining C4.4 tests.
- P5 · C5.1 interpolate/fit/L2/QI (staged callable API §3.6), the rest of extraction
  (_bspline_extraction, multilevel pieces not needed by THB in S6).
- P6a · C6.1 grids/BVH/tags/cell_quadrature/overlay/Partition/partition_grid (R1: Grid
  closed to subclassing, test_grid_abc rewritten), C6.2 locate, their tests.
- P6b · C6.3 THB/local space/coupling graph (stale-grid check in the wrapper §3.2), C6.4
  spectral partition on Eigen/Spectra + METIS C++ provider + pymetis path, tests.
  P6b.1 = **B1** spectral partition picks a non-Fiedler vector above 512 vertices
  (`bspline/_partition_graph.py:275-278`, measured during the design): failing regression test
  first, then the deflation fix, in its own commit.
- P7 · C7.1 cad + multipatch in C++ with wrappers and tests.
- P8 · MPI_THREAD_FUNNELED checked at entry (§2.8); `pantr[mpi]` = mpi4py +
  `pantr-mpi==same version`; C8.1 MPI algorithms/AnyCommunicator/capsule/MpiCommunicator/C++ MPI tests,
  C8.2 PyCommunicator/_pantr_mpi/pantr-mpi wheel/family check/thread policy/tests (U2, U5, U9 at
  start). Design note §13 (4).
- P3 · C3.1 BeRF-FR adapter in `providers/berf/` with its own LICENSE (exhaustive claim
  mapping), contract suite, pantr-berf wheel (Linux/macOS/Windows), `pantr[berf]` pulls
  `pantr-berf==same version`, `-DPANTR_WITH_BERF` OFF by default, wheel-content check (no BeRF-FR
  header in `pantr` or the default C++ build), CI leg 13, FetchContent from the public repo at a pinned SHA (R7). **If BeRF-FR is still
  private here, return to the maintainer.**
- P9 · C9.1 remove numba/threadpoolctl/_numba_compat/warmup/NUMBA env, AC5 import
  check, final full benchmark run, changelog and docs (install, C++ guide). Check: an independent read
  of the diff + full CI.
- P10 · the end review, proposed to the maintainer (reviewers, regions, gaps) before it runs: a
  review of the whole pull request across numerical correctness, tolerances (the provenance of
  every tolerance the port adds or changes, including the frame of the root-finding default
  tolerance), concurrency under TSan, public API and changelog, packaging and tests; then fixes, a
  final review gate, and a verification of AC1-AC13.

## Dependencies and licences (AC12)
Eigen 5.0.1 or 3.4.1 (MPL-2.0), Spectra 1.2.0 (MPL-2.0), Kokkos mdspan 0.6.0 (Apache-2.0 WITH
LLVM-exception), Boost.Multiprecision standalone (BSL-1.0), nanobind exact pin (BSD-3), doctest
(MIT, tests), scikit-build-core (Apache-2.0, build), numpy/scipy/mpi4py (BSD-3), pymetis (MIT, to
confirm), METIS (Apache-2.0, user-provided), BeRF-FR (pantr-berf only), MPI library (MPICH / Open MPI
permissive, Intel MPI proprietary-redistributable; user-installed, never bundled; confirm in S8).

## Proceeds unless told otherwise (QB)
QB1 C++ public API validates and throws everywhere; kernels unchecked. · QB2 limit_blas removed.
QB4 nanobind 3.1.x pinned (fallback 2.14.x if S0 fails). · QB5 Eigen 5.0.1 if Spectra passes on it.
QB6 names pantr-mpi/pantr_mpi, pantr-berf/pantr_berf. · QB7 Bezier clipping depth cap warns
(RuntimeWarning) instead of truncating silently. · QB8 pantr[metis] keeps pymetis. · QB9
Boost.Multiprecision for exact rationals. · QB10 new public API set_root_provider/get_root_provider
+ PANTR_ROOT_PROVIDER, default berf when installed. · QB11 1 pool thread per MPI rank by default.

## Inferred directives for a green light
- Di1 Every existing `match=` regex keeps matching (DR-13).
- Di2 Python wrappers over private nanobind classes; `pantr._testing` is the only private surface
  tests may import; tests importing private Numba kernels are re-pointed (no shim keeps old paths).
- Di3 Pickling and copy become explicit (`__reduce__`, `__copy__`, `__deepcopy__`) so they are
  not lost silently.
- Di4 The duck-typed communicator path of pantr.mpi stays supported (PyCommunicator), not test-only.
- Di5 **Deviates from R0's letter** ("Kokkos only where `<mdspan>` is missing"): a configure-time
  probe of `<mdspan>` sets `PANTR_USE_STD_MDSPAN` on the exported target, so every TU of a program
  sees one type (a header `__has_include` test would give two types in a program mixing C++20 and
  C++23 TUs, an ODR violation, §2.6). Same intent, decided at configure time.
- Di6 Development builds install the extension into src/pantr with `make ext`; no editable install.

## Changelog entries expected (AC9), each with its reason
- Python >= 3.12: R0 (numpy/scipy floors, abi3 cp312).
- Wheels need macOS >= 13.4: floating `std::to_chars` (DR-8).
- Grid cannot be subclassed from Python: R1 (C++ grids have no runtime polymorphism).
- Constructors copy; control_points and numeric views read-only; reverse() no longer a view; arrays
  taken before an in-place operation keep old values: R5 (immutable C++ values).
- Constructors' copy term exempt from the no-slower criterion: R9 (internal, noted for users).
- `limit_blas` removed (pantr.num_threads, mpi.configure_threads): R0 removes threadpoolctl.
- `NUMBA_NUM_THREADS` no longer read: max threads = pool capacity, override `PANTR_NUM_THREADS`, and
  the env var that disables pantr.mpi's default thread policy becomes `PANTR_NUM_THREADS`: R0
  removes Numba. The two tests matching "NUMBA_NUM_THREADS" (test_parallel.py:45,
  test_mpi_thread_policy.py:159) are AC1 category 3 (replaced Numba-specific tests).
- Root-provider switch `set_root_provider`/`get_root_provider`/`PANTR_ROOT_PROVIDER` (new): R0's
  runtime switch for BeRF-FR.
- Pickling and copying explicit: needed to keep them working over nanobind handles (Di3).
- Bezier root finding warns when its depth cap fires instead of truncating silently (QB7).
- New distributions pantr-mpi and pantr-berf: R0.
- Spectral partition picks the true Fiedler vector above 512 vertices (B1): a fix, not forced by C++;
  partitions of large graphs change. Approved with the plan (R11).

## Later
Smoothing/penalized fits, node layout as a type, THB L2 per level, AD Dual, requested minimum
root claim, curve-curve intersection, subset extraction API, pantr-metis wheel, split-mode and
free-threaded builds, cgroup-aware pool sizing, distributed L2 on owned cells only, explicit SIMD
kernels, precompiled instantiation, musllinux, Windows pantr-mpi, speed-ups beyond parity.

## DR records (from the review of the draft plan, and rulings R9, R10)
- DR-1 (R9) Constructors and copy=True conversions are exempt from AC6 for their O(n) copy term only;
  `copy=False` conversions share an immutable `shared_ptr<const Buffer>`, so the four
  `np.shares_memory` tests keep passing (test_bspline_conversion.py:636,692; test_bezier.py:164,178).
- DR-2 Coexistence: a Numba kernel is deleted in the stage that ports its last caller; each stage's
  commit group lists the surviving kernels. Unported Python callers of ported kernels go through an
  interim private surface `_pantr._kernels` (deleted in S9). Spanwise extraction moves into S4.
  `set_num_threads` drives both Numba and the C++ pool until S9; `test_jit_warmup_barrier` and the
  barrier stay until S9.
- DR-3 AC6 protocol: every gate measures main (its own checkout and env) and the branch interleaved
  in the same session, alternating processes, K = 10; a flag counts only if a fresh interleaved
  re-run flags it again. Statistic: branch median of process medians <= main median + main's range.
  §9.4 reports abi3-vs-version-specific separately from wrapper-vs-main; R6's split-mode trigger is
  the first delta only.
- DR-4 Communicator concept has one member per payload that exists today (`allgather_owned(dofs,
  coeffs)`, `allgather_values`, `allreduce_sum` with shape), one Python call each in PyCommunicator;
  the capsule vtable is noexcept with status codes, states GIL rules, holds a strong reference to the
  mpi4py Comm. U2 (read the 7 fake-comm test files) moves to P0a.
- DR-5 Pickling through state constructors (`_from_state(...)`), with a round-trip test per wrapper
  covering a refined HierarchicalGrid and a stale THB space; deepcopy keeps refinement.
- DR-6 MPI family check before loading `_pantr_mpi`: the build writes `pantr_mpi/_build_info.py`,
  compared with `mpi4py.MPI.Get_library_version()` first.
- DR-7 `EIGEN_DONT_PARALLELIZE` on `pantr::pantr` (never EIGEN_GEMM_THREADPOOL).
- DR-8 macOS deployment target >= 13.4 (floating `std::to_chars`), changelog line; arm64 Linux on
  `ubuntu-24.04-arm`, not QEMU; a scratch branch may trigger single legs for MSVC/macOS iteration
  (the port branch still runs full CI on every push).
- DR-9 `-march=native` decided in `pantrConfig.cmake` from a consumer-settable variable, not frozen
  at pantr's configure time (still optional, R0).
- DR-10 Pool is fork-safe: pid checked on submit, rebuilt in the child; a test in a forked child.
- DR-11 Identity of operation results: a holds-accessor returns the same rep when unchanged; the
  wrapper reuses the input's wrapper iff the rep pointers are equal (test_transform.py:619).
- DR-12 CellTags/FacetTags are immutable C++ values whose `set` returns a new value; the grid wrapper
  holds them as a separate handle beside `_h` and swaps that one, so setting a tag never swaps `_h`
  and THB spaces never go stale from it; the snapshot stays keyed on (max_level, num_cells, version).
- DR-13 Messages: the oracle is "every existing `match=` regex still matches" (replaces Di1's
  "verbatim"); C++ messages carry the matched fragments; Python-specific reprs (dtype('float64'))
  stay in wrapper checks.
- DR-14 Worktree subprocess tests and lint-imports run with an absolute PYTHONPATH and assert
  `pantr.__file__` is under the tree and `_pantr` imports (P0a), since the env's editable pantr
  points at the main checkout.
- DR-15 An `mpi-capsule` CMake component without Eigen/Spectra/mdspan for the pantr-mpi sdist.
- DR-16 Configure message corrected: before GCC 14, `-ffp-contract=on` behaves as `off`
  (gcc-13.3.0 manual); contraction is active on GCC >= 14 and Clang.
- DR-17 The AC1 ledger of changed assertions per stage is generated by a script diffing test
  assertions, not by hand (P0a).
- DR-18 (R10) pantr-cpp is made public before the first push of the branch (end of S0), after a sweep
  of the commits not in FELIGN/pantr for private names; P0a's first commit adds the branch to the CI
  push trigger so that first push runs full CI.
- DR-19 (mechanical) "pantr._parallel on C++" moves from P0a to P0b: it needs the
  extension module, which P0b's packaging builds. P0a's pool is C++-only, tested with ctest + TSan.
