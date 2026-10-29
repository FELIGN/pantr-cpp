# pantr C++ port: architecture

The architecture of the C++ port, approved with the plan (`design/port_plan.md`). The claims it
rests on were checked independently before approval (section 18), and the results are folded into
the sections they touch. Identifiers: R0-R11 (rulings), AC1-AC13 (acceptance criteria), DR-n
(decision records), Di1-Di6 and QB1-QB11 (QB3 unused) are defined in `design/port_plan.md`; M1-M7 (mirror tests),
U1-U9 (unverified items) and V1-V12 (verified claims) are defined here. "Lesson 1", "lesson 2" and
"lessons 3.n" are sections 1, 2 and 3.n of `design/lessons_first_port.md`, what a first port attempt
learned. Where this design was later
amended in the build, the amendment is recorded in the plan or in the short notes beside this file.

Source state read: pantr-cpp `main` 11ab436; BeRF-FR `b865e64a` (its build scheme as a pattern only,
no code copied). Where this design deviates from the letter of a ruling, it says so and why
(section 15).

## 0. The decision in one page

1. **Two layers, one of them public in each language.** A header-only C++20 library (`pantr::pantr`, `pantr::mpi`)
   that owns every domain type and validates its public API in every build type, and a Python package whose
   public modules are **thin Python wrapper classes over private nanobind classes** (`pantr._pantr`). nanobind classes
   are never the public Python types. Reason: dtype dispatch (two C++ instantiations behind one Python class), the
   handle swap R5 needs, `@overload`ed returns, union-typed inputs, Google docstrings, doctests and strict mypy all
   live naturally in Python and badly in binding code.
2. **Domain types are immutable values with a shared representation**: `class Bspline<T> { std::shared_ptr<const
   detail::BsplineRep<T>> rep_; }`. Copy is a refcount bump; derived quantities are computed eagerly when cheap, else
   in one grouped lazy block per type (double-checked locking on `std::atomic<bool>`). Python mutation (`in_place=True`,
   `HierarchicalGrid.refine`) swaps the value the wrapper holds.
3. **Kernels are generic over the scalar** (`template <class Real>`), domain types over the storage type
   `T in {float, double}`, and every site that computes in float64 today keeps doing so (dtype is an output format
   only where it is one today).
4. **One thread pool, deterministic chunking.** Workers sized from the affinity mask; the parallelism of a call is the
   calling thread's setting (thread-local, as Numba's is); chunk boundaries depend on the problem size and a derived
   grain, never on the thread count, and no parallel loop reduces across chunks. Consequence: results do not move with
   the thread count, and the warmup barrier and serial twins disappear.
5. **Errors are exceptions; incompleteness is a status.** A `pantr::error` hierarchy deriving from the standard
   exceptions, translated explicitly to Python types; messages ported verbatim because 852 `pytest.raises(...,
   match=...)` assert them. A spent budget or a dropped search region is a status field in the C++ result, which the
   Python wrapper turns into today's `RuntimeWarning`.
6. **Root finding is a seam with a pantr-owned result contract**: a `RootProvider` concept, a built-in provider ported
   from the Python solver, a BeRF-FR adapter in `providers/berf/` built as `pantr::berf` and the `pantr-berf` wheel.
   Each result carries a pantr-owned level (`computed`, `certified`, plus completeness); the built-in provider emits
   `computed` only. The Python switch is at the public function, so no C++ object crosses extension modules.
7. **MPI through a communicator concept.** Distributed algorithms are templates over a `Communicator` concept and do
   not include `mpi.h`; `pantr::mpi` adds `MpiCommunicator` (the only header that includes `mpi.h`). In Python, the
   algorithms are compiled once into `_pantr` behind a type-erased communicator; `_pantr_mpi` is a tiny extension that
   turns an mpi4py communicator into that type-erased handle. The duck-typed fake communicators of the suite reach the
   same C++ algorithms through a `PyCommunicator` adapter, so the suite stays the oracle.
8. **Packaging**: three distributions from one repo: `pantr` (core, multi-ISA, abi3 cp312, ships its own headers and
   CMake config), `pantr-mpi` (only `_pantr_mpi`, one MPI family per platform, dolfinx scheme), `pantr-berf`
   (adapter + BeRF-FR, multi-ISA). Arrays, never C++ objects, cross between extension modules.
9. **Dense linear algebra on Eigen** with numpy's decision rules (pivot-exactly-zero, SVD cutoffs) reproduced as named,
   cited constants, so the discrete outcomes the suite asserts keep their meaning.
10. **Ten stages**, foundation first, each green on full CI; Numba coexists with C++ until the last stage removes it.

## 1. Repository and CMake layout

### 1.1 Tree

```
CMakeLists.txt                  # root: project(pantr VERSION from include/pantr/version.hpp), options, add_subdirectory
cmake/
  PantrOptions.cmake            # every option, one place
  PantrDependencies.cmake       # FetchContent, full-SHA pins, SYSTEM, EXCLUDE_FROM_ALL
  PantrCompilerProbes.cmake     # feature probes (concepts, FP to_chars, <mdspan>), never version tests (one exception, 1.3)
  PantrFlags.cmake              # pantr::cxx_warnings (not exported), contraction, native arch
  PantrInstall.cmake            # install(EXPORT), config, version file
  pantrConfig.cmake.in          # find_dependency(Threads/Eigen3/Spectra/mdspan); components: mpi, berf, metis
include/pantr/                  # THE public C++ API, header-only (pantr::pantr)
  pantr.hpp                     # umbrella, never includes mpi.h
  version.hpp  config.hpp       # config.hpp: #error on __FAST_MATH__ / __FINITE_MATH_ONLY__
  core/        errors.hpp format.hpp ndarray.hpp mdspan.hpp lazy.hpp simd.hpp exact_rational.hpp
  parallel/    pool.hpp parallel_for.hpp affinity.hpp work_model.hpp
  linalg/      dense.hpp (thin Eigen front: solve, lstsq, svd_truncated, qr_solve, pinv, inv, eigh)
  tolerance.hpp
  quad/ basis/ change_basis/ geometry/ transform/ bezier/ roots/ bspline/ grid/ thb/ extraction/ partition/ cad/ multipatch/
  mpi/
    communicator.hpp            # concept Communicator + AnyCommunicator (type-erased, MPI-free)
    algorithms/*.hpp            # distributed space, collocation, l2, qi, thb_qi, from_partition: MPI-free templates
    mpi_communicator.hpp        # pantr::mpi target only: the ONE header that includes <mpi.h>
    comm_capsule.h              # C ABI struct (versioned) shared by _pantr and _pantr_mpi
  detail/  kernels/             # internal; no validation; Note: in every docstring
providers/
  berf/  LICENSE  CMakeLists.txt  include/pantr/providers/berf/provider.hpp  python/ (pyproject, pantr_berf/)  tests/
  metis/ include/pantr/providers/metis/partitioner.hpp   # C++ only, -DPANTR_WITH_METIS=ON
bindings/
  CMakeLists.txt                # _pantr (multi-ISA), _pantr_cpu (baseline probe)
  src/ module.cpp  <one .cpp per Python subpackage>  ndarray_io.cpp  errors.cpp  testing.cpp  py_communicator.cpp
  mpi/ CMakeLists.txt module.cpp pyproject.toml (pantr-mpi) pantr_mpi/__init__.py
src/pantr/                      # Python package (wrappers), unchanged public layout
  _ext.py                       # loads the ISA rung, interface-version and staleness checks
  _isa.py                       # the ladder (shared with pantr-berf)
  _testing.py                   # re-exports _pantr._testing: the single private surface tests may import
  _pantr*.so  _pantr_cpu*.so    # built in place for development (gitignored), installed in wheels
tests/                          # Python suite (the oracle), unchanged layout
  oracles/                      # shared independent oracles (exact rational, closed form), section 8.3
cpp/tests/                      # doctest suite over the public headers; consumer/ (find_package smoke project)
benchmarks/                     # harness + cases, section 9
design/                         # short notes for the expensive-to-reverse decisions, section 13
```

`cpp/` holds only tests and the consumer project; public headers sit at the repo root's `include/` so the include
path in the build tree and in the install tree is the same string.

### 1.2 Targets

| Target | Kind | Carries | Exported |
|---|---|---|---|
| `pantr::pantr` | INTERFACE | `include/`, `cxx_std_20`, `Threads::Threads`, `Eigen3::Eigen`, `Spectra::Spectra`, `std::mdspan`/Kokkos `mdspan::mdspan`, contraction flag, optional `-march=native` (added in the consumer, DR-9), `EIGEN_MPL2_ONLY`, `EIGEN_USE_AVX512_TRSM_KERNELS=0` (GNU) | yes |
| `pantr::mpi` | INTERFACE | links `pantr::pantr` + `MPI::MPI_CXX` | yes, component `mpi` |
| `pantr::berf` | INTERFACE | `providers/berf/include`, links `pantr::pantr` + `berf::berf` | only with `PANTR_WITH_BERF=ON`, component `berf` |
| `pantr::metis` | INTERFACE | `providers/metis/include`, links `METIS::metis` | only with `PANTR_WITH_METIS=ON` |
| `pantr_cxx_warnings` | INTERFACE, private | `-Wall -Wextra -Wconversion -Wsign-conversion -Wshadow -Wdouble-promotion [-Werror]` | no |
| `_pantr`, `_pantr_v3`, `_pantr_v4` (x86) / `_pantr`, `_pantr_neon` (arm) | nanobind modules | `NB_STATIC NOMINSIZE NB_SUPPRESS_WARNINGS STABLE_ABI` | wheel only |
| `_pantr_cpu` | nanobind module | baseline, no pantr headers | wheel only |
| `_pantr_mpi` | nanobind module, `bindings/mpi` | MPI glue only, no numerics | `pantr-mpi` wheel |
| `_pantr_berf[_v3/_v4/_neon]` | nanobind modules | adapter + BeRF-FR | `pantr-berf` wheel |

### 1.3 Options and flags

| Option | Default | Effect and reason |
|---|---|---|
| `PANTR_FP_CONTRACT` | ON | `-ffp-contract=on` (`/fp:contract`) on `pantr::pantr`, inherited by consumers. On GCC < 14, where `on` behaves as `fast` (lesson 1, row "Never -ffast-math"), set `-ffp-contract=off` instead and say so in a configure message. This is the one version test: contraction semantics cannot be probed by compiling a snippet. |
| `PANTR_NATIVE_ARCH` | ON | `-march=native` for the consumer, as ruled; decided in `pantrConfig.cmake` from the consumer's own setting, never frozen into the export (DR-9) (chosen by the maintainer after the portability warning). Forced OFF in every wheel build and every CI leg that targets a named ISA. |
| `PANTR_FORCE_SCALAR` | OFF | defines `PANTR_SIMD_FORCE_SCALAR`. |
| `PANTR_USE_STD_MDSPAN` | OFF | see 2.6; picks `std::mdspan` for every consumer TU uniformly. Not supported in stage 0 (`design/mdspan_identity.md`). |
| `PANTR_WITH_BERF` / `PANTR_WITH_METIS` | OFF | build, test and export the provider target. Never changes a core header. |
| `PANTR_BUILD_TESTS` / `PANTR_BUILD_PYTHON` / `PANTR_BUILD_PYTHON_MPI` / `PANTR_BUILD_BENCHMARKS` | ON (top level) / OFF / OFF / OFF | as named. |
| `PANTR_MULTI_ISA` | OFF | the per-ISA module variants (wheels). Fatal with `PANTR_NATIVE_ARCH=ON`. |
| `PANTR_USE_SYSTEM_DEPS` | OFF | `find_package` instead of FetchContent; OFF by default because `FIND_PACKAGE_ARGS` once picked a system Eigen of another major version silently (lesson 2). |
| `PANTR_WARNINGS_AS_ERRORS` | OFF (ON in CI) | on `pantr_cxx_warnings` only. |
| `PANTR_INSTALL` | ON | install headers, export, config; the bundled third-party headers under `include/pantr/third_party/` when fetched. |

`config.hpp` has `#if defined(__FAST_MATH__) || defined(__FINITE_MATH_ONLY__) #error`. This puts the fast-math
prohibition in a channel that runs on every consumer compile, which a configure-time check of `CMAKE_CXX_FLAGS`
cannot do for a header-only library (*state the floating point flags*).

### 1.4 Dependencies (FetchContent, full SHA, `SYSTEM`, `EXCLUDE_FROM_ALL`)

Eigen (5.0.1 by default if Spectra's contract tests pass against it in stage 0, which a single verification probe
suggests; else 3.4.1), Spectra 1.2.0 (or the first master SHA after its 2025-08-26 fix for an initial vector in the null space, if its diff
is relevant once read in stage 6), Kokkos mdspan (mdspan-0.6.0), Boost.Multiprecision standalone (exact rationals, 2.8), doctest (tests), nanobind (exact pin,
build-system), BeRF-FR (pinned SHA, `PANTR_WITH_BERF` only). Offline: `FETCHCONTENT_SOURCE_DIR_<DEP>` and
`FETCHCONTENT_FULLY_DISCONNECTED`; `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` for CMake 4 against old dependency floors
(lesson 2). Licences: section 11.

### 1.5 Install, export, config (BeRF-FR has none; pantr must)

`install(TARGETS pantr EXPORT pantrTargets)`, `install(DIRECTORY include/pantr)`,
`install(EXPORT pantrTargets NAMESPACE pantr:: DESTINATION lib/cmake/pantr)`, `pantrConfigVersion.cmake` with
`SameMinorVersion` (lesson 2). `pantrConfig.cmake` does `find_dependency(Threads)` (headers use `std::mutex`),
`find_dependency(Eigen3)`, `find_dependency(Spectra)`, `find_dependency(mdspan)` (unless `PANTR_USE_STD_MDSPAN`),
and per component `find_dependency(MPI COMPONENTS CXX)` (`mpi`), `find_dependency(berf)` (`berf`). When the build
fetched the dependencies, `PANTR_INSTALL` also installs their headers and config files under the same prefix, so
`cmake --install` of pantr alone yields a prefix a consumer can use with no other package (AC4). The contraction flag
is in the exported target; the warnings target is not.

**The Python wheel ships pantr's own headers and CMake config** (`pantr/include`, `pantr/share/cmake/pantr`, and
`pantr.cmake_dir()`), as pybind11 and nanobind do, but not the third-party headers. This is what lets `pantr-mpi` and
`pantr-berf` be built from their sdists against an installed `pantr` (their CMake fetches the pinned third-party
dependencies itself): the advanced-user path of R0 (`pip install --no-binary pantr-mpi pantr-mpi` against a local MPI).

### 1.6 `providers/berf/` and the `pantr-berf` wheel

`providers/berf/LICENSE` (the adapter's own licence, as ruled), `include/pantr/providers/berf/provider.hpp`
(`pantr::berf::Provider`, section 5.1), `tests/` (the provider contract suite instantiated for it), `python/` with its
own `pyproject.toml` (scikit-build-core, name `pantr-berf`, `requires = pantr==<same version>` at build and run
time, nanobind pinned to the same exact version), installing top-level package `pantr_berf` with
`_pantr_berf[_v3|_v4|_neon]` and reusing `pantr._isa` for its ladder. `pantr[berf]` is an extra pulling
`pantr-berf==<same version>`. The `pantr` wheel's CMake never adds `providers/berf/include` to any target; AC7's content
check greps the wheel for `berf` headers. The legal soundness of a mixed-licence monorepo stays an open item the
maintainer has flagged, not something this design settles.

### 1.7 `_pantr_mpi` and the `pantr-mpi` wheel

`bindings/mpi/` builds `_pantr_mpi` (top-level package `pantr_mpi`), which links MPI and contains only: the mpi4py
communicator conversion (`PyMPIComm_Get`), an `MpiCommunicator` wrapped into the `comm_capsule.h` C struct, the MPI
family it was built against (from `MPICH_VERSION` / `OMPI_MAJOR_VERSION` / `I_MPI_VERSION`), and `MPI_Query_thread`.
No numerics, so no ISA variants. Wheels follow dolfinx's scheme as surveyed before this design (FEniCS/dolfinx#4652, #4570;
confirm in stage 8 because the scheme moves): Linux built against the MPICH ABI, macOS against Open MPI, MPI library
not bundled, `requires mpi4py`; the MPI runtime is installed by the user (`mpich`, `openmpi`, `impi-rt` wheels or a
system MPI). `pantr[mpi]` = `mpi4py` + `pantr-mpi==<same version>`. Import-time check in `pantr.mpi`: the family
string `_pantr_mpi` was built for must match the family of `mpi4py.MPI.Get_library_version()` (dolfinx's own check,
verified at #4652's merge commit be8352e, is family-level, lives only in built wheels by template injection, and on
Linux tests "not Open MPI"; it misses an MPI offering only the MPI-5 standard ABI. pantr records the family in
`_pantr_mpi` itself, so source and wheel builds share one check); a missing libmpi
(loader error) and a mismatch both raise `ImportError` naming expected and loaded (AC10). Advanced users build
`pantr-mpi` from its sdist against any MPI (`MPICC`/`CMAKE_PREFIX_PATH`), which disables the family check by
recording `family=custom`.

### 1.8 Multi-ISA build and ladder

Copy BeRF-FR's pattern (not code): one source set compiled per ISA into differently named modules
(`NB_MODULE(PANTR_EXT_MODULE_NAME, m)`): x86 `_pantr` (x86-64 baseline), `_pantr_v3` (`-march=x86-64-v3`),
`_pantr_v4` (`-march=x86-64-v4`); arm64 `_pantr` (forced scalar) and `_pantr_neon` (NEON baseline; whether a separate
rung pays on AArch64 is measured in stage 0, BeRF-FR ships arm64 single-module). `_pantr_cpu` is always built at the
toolchain baseline with no pantr headers; `supports("v3"|"v4")` checks CPUID **and OS-enabled vector state (XGETBV)**,
fails closed (lesson 2). `pantr/_isa.py`: `PANTR_ISA` set means exactly that rung, a missing rung raises (no
fallthrough, so a benchmark cannot measure the wrong module); unset means the best supported rung, then the base.
The loaded module exposes `isa` so the choice is observable. One rule this layout imposes on the C++ code: **no
public type's layout may depend on ISA macros or on Eigen fixed-size vectorizable members**, so a C++ consumer mixing
TUs compiled for different ISAs (function multiversioning) cannot hit an ODR layout mismatch.

### 1.9 Python build and the development loop

`pyproject.toml`: `build-backend = scikit_build_core.build`, `requires = ["scikit-build-core==<pin>", "nanobind==<pin>"]`,
`wheel.py-api = "cp312"`, `install.strip = false` and `NOMINSIZE` (the `-Os`-after-`-O3` trap, a silent factor of
about three, lesson 2), `build-dir = "build/skbuild/{wheel_tag}"`, `cmake.define.PANTR_NATIVE_ARCH = "OFF"`,
`requires-python >= 3.12`, `numpy` and `scipy` runtime, Numba and threadpoolctl removed in the last stage.

Development uses **no editable install**: `make ext` configures `build/dev` and installs the modules into `src/pantr/`
(gitignored), so `pytest.ini`'s `pythonpath = src` keeps working per worktree and the `ScikitBuildRedirectingFinder`
shadowing trap (lesson 2) cannot occur. Two guards make a stale module loud instead of a phantom regression (lesson 2):
`_pantr.interface_version` must equal `pantr._ext.EXPECTED_INTERFACE` (bumped with any binding change), and the test
session's `conftest.py` fails when the module is older than any file under `include/`, `bindings/` or
`providers/*/include`. CI builds the same way, and a separate wheel job tests the installed wheel with
`-o pythonpath=`. `make stubs` regenerates `src/pantr/_pantr.pyi` with `nanobind.stubgen`; the committed stub is what
mypy reads, and the tests job fails on a stub diff.

## 2. Core types

### 2.1 Ownership classes (lesson 1, row 1: reused, now with one backend)

Each accessor returning domain data is one of three kinds, stated in its docstring:

- **constructs**: returns a new value (`Bspline::reversed()`, `Bezier::split(t)`, `BsplineSpace1D::insert_knots`).
- **holds**: the owner stores a value of another domain type and returns it by value (cheap: a refcount bump)
  (`Bspline::space()`, `THBSplineSpace::grid()`).
- **numeric view**: returns a read-only view (`std::span<const T>` / `mdspan<const T>`) into the owner's
  representation; the Python binding wraps it as a read-only `ndarray` whose capsule owner holds a copy of the
  `shared_ptr`, so the array outlives a handle swap and stays correct (it shows the old value, see 3.2).

No `reference_internal` anywhere in the binding; a refcount assertion in the C++ tests detects a broken lifetime.

### 2.2 Immutability and derived state

All public domain types are immutable after construction (R5); `operator=` exists (values), but no member mutates.
Representation: `std::shared_ptr<const detail::XRep<T>>`. Derived state:

- **eager** when it costs no more than validation (unique knots and multiplicities, first basis per interval, domain,
  open-end flags of `BsplineSpace1D`; this retires the module-global `lru_cache` keyed on knot bytes).
- **lazy** otherwise, as one grouped block per type behind `detail::Lazy<Block>`: `std::atomic<bool>` acquire load,
  `std::mutex` on the slow path, release store (lesson 1, row 3; not a bare `mutable std::optional`, which races and
  passes value tests; not `std::call_once`). Members: `Bspline` (Bezier decomposition, locate context incl. BVH),
  `THBSplineSpace` (truncation coefficients, contribution cache, `max_active_per_cell`), `Grid` (BVH), extraction
  operators.
- **process caches**: only for pure functions of small integer keys with a declared bound: change-of-basis matrices
  and degree-reduction operators keyed on `(kind, degree[, decrement], scalar)`, mutex-guarded, bounded by the
  per-dtype degree limits that already exist. This narrows lesson 1's "no process-global cache": the lesson's hazard
  (unbounded growth, keys on domain objects) does not apply to a table indexed by `degree <= 26`, and dropping the cache
  would cost AC6 on repeated `change_basis` calls. Each such table is a TSan target.

Every lazy block and process table has a TSan test that races N threads on first access (AC3).

### 2.3 Scalar genericity (R3) and dtype

- Kernels: `template <class Real>` using unqualified math with `using std::sqrt;` etc., `value_of(x)` for
  comparisons, so `float`, SIMD lanes and a later `Dual` instantiate (lesson 1, AD row).
- Domain types: `template <class T> requires std::same_as<T,float> || std::same_as<T,double>`; Python binds
  `BsplineF32`/`BsplineF64` and the wrapper picks by dtype.
- Compute precision follows today's code site by site, not a new global rule: float32 objects evaluate in float32 where
  Numba did; locate, roots, QI, degree reduction, interpolation solves compute in float64 and cast back, as the Python
  does. Quadrature node generation computes in double and stores T (lesson 1: a float Newton Gauss-Legendre loses
  weight accuracy). Kernels with a data-dependent output length return the count.

### 2.4 Arrays at the C++ API

- Inputs: `std::span<const T>` for 1-D, `pantr::mdspan<const T, dextents<std::size_t, R>>` (row-major) for N-D.
- Outputs: every evaluation entry point has an **`out` overload** taking a caller-allocated mdspan (no allocation on a
  hot path, *hot path shape is a performance decision*) and a convenience overload returning
  `pantr::NdArray<T>` (owning, contiguous, runtime extents). The binding hands an `NdArray` to NumPy without a copy
  (capsule-owned buffer; V3).
- Runtime dimension everywhere (`dim` and `rank` are runtime; no new cap is introduced, the existing
  `_EINSUM_MAX_DIM = 24` in THB stays as it is).

### 2.5 Validation and errors (QB1)

Public C++ functions validate everything and throw; `detail/kernels` validate nothing and say so (`Note:` in the
docstring, as today). Debug builds add `PANTR_ASSERT` inside kernels (on in the sweep leg).

```
pantr::error                      (mixin base, carries nothing but identity)
pantr::value_error  : std::invalid_argument, error   -> ValueError
pantr::type_error   : std::invalid_argument, error   -> TypeError
pantr::index_error  : std::out_of_range,     error   -> IndexError
pantr::state_error  : std::logic_error,      error   -> RuntimeError   (e.g. a stale THB space, raised in Python today)
pantr::not_implemented : std::logic_error,   error   -> NotImplementedError
pantr::capacity_error  : std::length_error,  error   -> ValueError     (implementation limits: bincoeff envelope, dim > 24)
std::overflow_error -> OverflowError, std::bad_alloc -> MemoryError
```

The binding registers **one explicit translator** for the whole hierarchy instead of relying on nanobind's defaults.
Python-only errors (`KeyError`, `AttributeError`, `FrozenInstanceError`, `TypeError` for a wrong Python type, `ImportError`)
stay in the wrappers. **Messages are ported verbatim**: a test asserting `match=` is part of the oracle. `core/format.hpp`
provides `py_repr(double)` (shortest round trip, Python's `repr` spelling: `3.0`, `1e-05`, `inf`, `nan`) and
`py_repr(shape)` (`(3, 4)`), because some messages interpolate values. A message that cannot be reproduced exactly is
an AC1 change and is listed.

### 2.6 mdspan

C++20 has no `std::mdspan`; GCC 14.4's libstdc++ provides none even in C++23 mode, and Clang 18 on libstdc++ none
either (V1); libc++ has one. `pantr::mdspan` is
**Kokkos mdspan unless `PANTR_USE_STD_MDSPAN=ON`**, a compile definition set on the INTERFACE target so every consumer
TU agrees. The ruling says "Kokkos only where `<mdspan>` is missing"; implemented literally as an `#if __has_include`
in a header, the alias would name two different types in a program that mixes C++20 and C++23 TUs (an ODR violation
in every signature that mentions it). The CMake option keeps the ruling's intent (std where the project opts in) with
one type per program. Element access uses `operator()` in pantr code (multi-index `operator[]` is C++23).

### 2.7 Exhaustion and incompleteness

A loop's cap stays (it is the only finiteness argument in floating point, *a loop carries a loud bound*) and its
exhaustion becomes a status the caller receives, never a plausible value:

- `bspline::find_roots` returns `RootSet<double>{roots, status}`; `status` in `{complete, budget_exhausted,
  stalled_at_last_iterate}` maps one to one onto today's two `RuntimeWarning` messages, which the Python wrapper keeps
  emitting verbatim (`test_bspline_roots.py` asserts them with `pytest.warns(..., match=...)`).
- Bezier clipping today drops work silently at three caps (`_MAX_STACK_SIZE = 256`, `max_roots = 3n+4`,
  `_CLIP_MAX_DEPTH = 64`; `bezier/_clipping_core.py:116, 152-294`). In C++ the stack is a `std::vector` (the cap was a
  Numba constraint, now gone), `max_roots` cannot bind after dedup (a degree-n polynomial has at most n distinct roots;
  pre-dedup the buffer grows), and the depth cap stays with a status. See QB7 for what Python does with that status.
- `[[nodiscard]]` on every result type.

### 2.8 Thread pool and parallel thresholds

- `pantr::parallel::Pool`: one process pool, created on first use, worker count = popcount of the affinity mask
  (`sched_getaffinity` on Linux, `GetProcessAffinityMask` on Windows, `hw.activecpu` on macOS which has no affinity),
  never `hardware_concurrency()`. Capacity can be lowered by `PANTR_NUM_THREADS` at process start.
- `parallel_for(n, grain, body)`: chunks of `grain` items, boundaries a function of `(n, grain)` only. A job is an
  atomic chunk counter; the **calling thread executes chunks of its own job**, so progress needs no free worker,
  concurrent callers from several Python threads are safe (the hazard the Numba warmup barrier guarded), and a
  `parallel_for` called from inside a worker runs inline (no nested oversubscription, no deadlock). The first
  exception thrown in a chunk is captured and rethrown on the caller after the join.
- Parallelism of a call = the calling thread's `thread_local` setting (default: pool capacity). This reproduces
  Numba's thread-local semantics, which `pantr.mpi`'s per-rank policy documents. `pantr.set_num_threads`,
  `get_num_threads`, `num_threads()` keep their API; the maximum becomes the pool capacity instead of
  `NUMBA_NUM_THREADS`; `limit_blas` is removed (QB2).
- No parallel loop reduces across chunks; every parallel kernel writes disjoint outputs. With chunking independent
  of the thread count, a result does not depend on the number of threads (stated as a property, still tested within a
  derived tolerance, *bit identity is not a criterion*). A future reduction must keep fixed chunk order.
- Thresholds: `grain = max(1, ceil(C_chunk / w))` and "run serially when `n * w < C_par`", with `w` a per-kernel work
  estimate in elementary operations derived from the algorithm (for example `(p+1)^2` per point for Cox-de Boor),
  `C_chunk` set so one chunk amortises one atomic claim plus a cache-line of output, `C_par` so a job amortises a
  futex wake-and-join. Those two constants are order-of-magnitude OS and cache facts, recorded as **admitted
  heuristics** at their definition with that reasoning (*a multiplier is a tolerance*), never fitted on one
  particular many-core machine. This replaces `_PARALLEL_MIN_NUM_PTS = 4096` and the serial-twin kernels.
- The GIL is released around any call that reaches the pool, and only then (same work model).
- MPI: `MPI_THREAD_FUNNELED` is sufficient because collectives are only called by the calling thread, outside
  `parallel_for`; `pantr::mpi` checks `MPI_Query_thread() >= FUNNELED` at entry and says that the caller must be the
  thread that initialised MPI.

### 2.9 SIMD

`core/simd.hpp`: `PANTR_SIMD_AVX512 / AVX2 / NEON / SCALAR` from target macros, `PANTR_SIMD_FORCE_SCALAR` override,
`PANTR_SIMD_ISA_NAME`. Policy (lesson 1, SIMD row): vectorise across items, never inside a recurrence; auto-vectorise
first; block before SIMD; the macros select block widths (lanes) for blocked loops. Explicit intrinsics only where a
benchmark shows auto-vectorisation fails, with a scalar reference the forced-scalar CI leg compares against.

## 3. Binding architecture

### 3.1 Thin Python wrappers over private nanobind classes

`pantr._pantr` binds C++ values (`BsplineF64`, ...) with minimal, flat methods taking already-normalised arguments
(contiguous arrays of the right dtype, ints, enums). Each public Python class keeps its module, name, signature,
docstring and `@overload`s, and holds one attribute `_h`. The wrapper does: `np.asarray`/dtype dispatch, union-typed
argument resolution (`nodes="greville" | PointsLattice | ...`, the `isinstance` ladders in the factories), `@overload`
return shapes, Python-only exceptions, warnings from statuses, `out=` handling (3.4), and calls to user callables (3.6).
It validates no values: that is C++'s, once.

Alternatives considered: (a) nanobind classes as the public types: rejected, because one Python class must front two
C++ instantiations, docstrings/autodoc/stubs/doctests of binding-defined classes cost more than they save, and every
union input would become binding C++; (b) Python keeps validation and C++ checks nothing: rejected by QB1 and by
*two layers only one is the api* (the C++ API is public too). The cost accepted: one extra Python frame per call;
stage 0 measures it against main (section 9.4).

### 3.2 R5's handle swap

- `Bezier.reverse(in_place=True)`: `self._h = self._h.reversed()`; returns `None`. Same for `transform`,
  `permute_directions`, and `Bspline`'s three, which also replace the Python space object they hold.
- `HierarchicalGrid.refine/refine_cells/coarsen/coarsen_cells`: `self._h = self._h.refined(...)`. The C++ grid
  carries `version()`: the new value has `version + 1` when the structure changed and the same version on a no-op
  (today's semantics; `_rebuild` bumps only on change).
- `THBSplineSpace`: the C++ space owns its grid value, so in C++ a space is never stale. The Python wrapper keeps a
  reference to the user's grid wrapper (so `space.grid` returns the live object, as today) and the snapshot
  `(max_level, num_cells, version)`; every method that raises today compares the user grid's current `_h` against the
  snapshot and raises the same `RuntimeError` with the same message. Behaviour is unchanged; the check moved from the
  data to the wrapper. `refine/refine_region/coarsen` return a new space with a new grid wrapper (today: `deepcopy`).
- Changelog consequences (R5, AC9): constructors copy (no aliasing of the caller's array); `control_points` and every
  numeric view are read-only; non-in-place `reverse()` is no longer a view; an array obtained before an in-place
  operation keeps the old values; writes through `control_points` raise.
- Python identity is kept where today's code or tests rely on it: a wrapper stores the Python objects it was built from
  (`b.space is b.space`, `DistributedFunction`'s `global_function.space is distributed_space.global_space`).
- `__copy__`/`__deepcopy__` on every wrapper share the immutable handle; `HierarchicalGrid` copies its wrapper so a later
  `refine` on one does not affect the other. `__reduce__` from constructor arguments on every domain wrapper (lesson 1,
  pickle row): today the classes are plain Python objects and therefore picklable (multiprocessing, `comm.bcast`); a
  nanobind handle is not, so without `__reduce__` the port would silently remove that.

### 3.3 Grid without user subclassing (R1)

Python `Grid` stays a class: the base of the two wrappers, with `__init_subclass__` refusing any other subclass
(`TypeError`, changelog). `isinstance(x, Grid)` keeps working. The 21 default methods become C++ (`GridBase<Derived>`
CRTP, no vtable, lesson 1) and are bound on each concrete class. Generic consumers (`cell_quadrature`,
`partition_grid` rcb, `grid_to_pyvista`, `mpi/_create`) are C++ templates over a `GridLike` concept, instantiated for both
grids, dispatched in the wrapper by type. `tests/test_grid_abc.py` is rewritten against the two concrete grids (R1).

### 3.4 `out=` and non-contiguous outputs

The binding accepts `out` as a C-contiguous, writable ndarray of the exact dtype and shape and validates with the
existing messages ("Output array has shape", "has dtype", "is not writeable"). A non-C-contiguous `out` (supported
today, `tests/test_out_writeback.py`) is handled in the Python wrapper: compute into a temporary, copy back, return the
caller's `out`, as `_reshaped_out` does now.

### 3.5 Overloaded returns, PointsLattice, scipy.sparse

- Returns that depend on the argument type (`find_roots` single vs batch, `slice`/`boundary` returning `Bezier |
  ndarray`, `in_place` returning `None`) are decided in the wrapper; C++ has one function per return type.
- `PointsLattice` and `QuadratureRule` are C++ values with wrappers; `pts_per_dir` returns read-only views.
- `THBSplineSpace.prolongation_to_sparse`: C++ returns CSR arrays (`indptr`, `indices`, `data`); the wrapper builds
  `scipy.sparse.csr_array` (decided before this design). `scipy` is imported only in wrapper modules that build a returned scipy object
  (AC5's import check lists them: the THB wrapper module, nothing else).

### 3.6 User callables

All are batched today (as mapped from the Python code). The C++ API offers two shapes per fitting function: a **staged** one
the binding uses (C++ returns a value describing what to evaluate: a lattice, a list of boundary-trace lattices, a flat
`(M, dim)` point set; the wrapper calls `func` once per item in Python; C++ consumes the values), and a convenience
template taking a C++ callable for C++ users. No callable crosses into C++ from Python, none runs inside a parallel
region, the GIL question does not arise (lesson 1, callbacks row). Return-shape and dtype casting rules stay as today.

### 3.7 What viz needs

Only public API (from the map of `pantr.viz`): `to_beziers()` returning an object array of `Bezier` (the wrapper builds it and
caches it, invalidated on handle swap), `control_points`, `evaluate`, `get_unique_knots_and_multiplicity`,
`get_greville_abscissae`, THB `grid`, `num_levels`, `num_basis_per_level`, `active_function_indices`, `level_space`,
`MultiLevelExtraction(..., target="bezier").operator/active_basis/num_elements`, `Grid.ndim/num_cells/cell_bounds`. All
kept; viz tests run in every stage (headless: no `.show()`).

### 3.8 The 135 private (module, name) pairs in 32 test files (plus those outside the core modules)

Per stage, each pair goes to one of three bins, recorded in the stage's commit message:

1. **Public route**: the test is re-pointed at the public API (most knot helpers, evaluation helpers).
2. **Internal worth testing directly** (`_bincoeff`, `_dedup_roots`, `_morken_reimers_roots`, `_clip_roots_core`,
   `_yuksel_roots`, reduction operators, `_block_bounds`, grid block helpers, tolerance formulas pinned by M4): bound
   in `_pantr._testing` and imported from `pantr._testing`, the **only** private surface tests may import. No shim keeps
   an old private module path alive.
3. **Numba-specific** (serial cores, warmup, `_PARALLEL_MIN_NUM_PTS`, NUMBA env): the test is replaced (8.4).

Private attribute reads of wrappers in grid tests (`g._bvh`, `g._cell_tags`, `g._blocks`, ...) become private
properties of the wrapper reading the C++ value (`_bvh` returns `None` until built). The downstream consumer imports
public names only (as last censused, 2026-09-16); the downstream re-census runs before every stage that
reshapes a symbol.

## 4. Seams

### 4.1 Root finding

```cpp
namespace pantr::roots {
enum class Claim : std::int32_t { none = 0, computed = 1, certified = 2 };    // per root; wire values pinned
enum class Completeness : std::int32_t { complete = 0, gap_localized = 1, gap_unlocalized = 2 };
struct Capabilities { Claim max_claim; bool batch; bool monotone; bool multiplicity; };
template <class T> struct Root { T x; Claim claim; std::int32_t multiplicity; /* -1: no claim */ };
template <class T> struct RootSet {
  std::vector<Root<T>> roots;              // sorted, distinct within the request's param_tol
  Completeness completeness;
  std::vector<std::pair<T,T>> gaps;        // intervals with no claim (gap_localized)
};
struct FindRootsOptions { double param_tol; double value_tol; };   // frames stated, see below
template <class P, class T> concept RootProvider = requires(const P p, std::span<const T> bernstein, FindRootsOptions o) {
  { p.capabilities() } -> std::same_as<Capabilities>;
  { p.find_roots(bernstein, o) } -> std::same_as<RootSet<T>>;
};
class Builtin;                              // Yuksel + clipping + dedup, ported
template <class P = Builtin, class T> RootSet<T> find_roots(const Bezier<T>&, FindRootsOptions, const P& = {});
}
```

- **What is certified, named in the input's terms** (*name the object you certify*): roots of the stored scalar
  Bernstein polynomial on `[0, 1]` (for a rational Bezier, of its first homogeneous component, as today). `certified`
  means the provider proved existence and uniqueness of a root in `[x - param_tol, x + param_tol]`; `computed` claims
  nothing beyond "returned by the algorithm".
- **Weakest producer** (a result is only as certified as the weakest step that produced it): `Builtin` emits only `computed`; its completeness is `complete` unless
  a cap fired (2.7). The BeRF-FR adapter maps BeRF-FR's per-row certificate and result level through **one exhaustive
  `switch` with no default** (*a closed set needs exhaustive dispatch*): a BeRF-FR certificate maps to `certified`
  only where its documented claim is existence-and-uniqueness for that row; everything else maps to `computed`; any
  BeRF-FR gap level maps to `gap_localized`/`gap_unlocalized`. A new BeRF-FR enum member fails the adapter's compile.
- **Capability query, not a requested tier** (lessons 3.6 open item): `capabilities()` is a value; requesting a minimum
  claim through the options is on the later list (no caller needs it, and a request the provider cannot honour would
  need its own failure mode).
- **Frames** (*a certified tolerance names its frame*): `param_tol` is absolute on the parameter interval `[0, 1]`;
  `value_tol` is absolute in the coefficient units. Today's default passes the dimensionless `get_strict(dtype)` into
  both (found while mapping the Python code). The port keeps the default bit-for-bit (the suite is the oracle) and states the frame at the
  options struct; whether the default should scale with the coefficient magnitude is routed to the end review's
  provenance check of tolerances, not changed here.
- **Python**: `pantr.bezier.find_roots/find_monotone_root` keep their signatures and return arrays. The wrapper calls the
  active provider (`pantr._pantr` or `pantr_berf`) with plain coefficient arrays; a result with
  `completeness != complete` emits `RuntimeWarning` (one shared wording with the B-spline path). Runtime switch:
  `pantr.bezier.set_root_provider("builtin"|"berf")`, `get_root_provider()`, initial value from `PANTR_ROOT_PROVIDER`,
  default `berf` when `pantr_berf` imports, else `builtin` (AC7's "switch both ways"; new public API, changelog).
  `find_roots(..., return_claims=True)`-style access to claims is on the later list.
- **Dedup** (merging duplicate roots must not merge distinct ones): the pair-classes of duplicates (adjacent subintervals sharing an endpoint, a
  multiple root found by two clip branches, a root at 0 or 1 found as an endpoint and by Newton) are tabulated against
  the layer that kills each in the `roots/` design note; the provider contract test asserts distinctness for each class.
- **Contract suite**: one templated C++ test suite over `RootProvider`, run for `Builtin` and (with
  `PANTR_WITH_BERF`) for `berf::Provider`: sorted, distinct, in `[0,1]`, claim not above `max_claim`, encoded-truth
  polynomials (products of known linear factors in Bernstein form, built exactly), each completeness value reached
  by a constructed input (*acceptance is a theorem completeness may be cheap*: assert which rung answered).

### 4.2 Graph partitioning

`template <class P> concept GraphPartitioner` with `partition(const CouplingGraph&, int n_parts, std::span<const
std::uint8_t> active) -> Partition`. Implementations: `Spectral` (default, recursive Fiedler bisection:
dense `Eigen::SelfAdjointEigenSolver` for `m <= _DENSE_FIEDLER_MAX`, as today; above it, Spectra `SymEigsSolver`
with `SortRule::SmallestAlge` for **one** eigenpair of the Laplacian **projected onto the complement of the constant
vector** (`x -> P L P x`, `P = I - 1 1^T / m`, the known null vector of a connected graph's Laplacian), with the same
centred `v0`, falling back to the dense solver when Spectra reports non-convergence; the fallback is recorded in the returned
`PartitionInfo`, so it is not silent), and `providers/metis` (`-DPANTR_WITH_METIS=ON`, C++ users).

Python `partition_graph(backend="metis")` keeps calling **pymetis** at the Python boundary (`pantr[metis]` unchanged),
handing it the CSR arrays of the C++ `CouplingGraph` and building `Partition` from the membership through C++. This is
exactly what exists and avoids a fourth compiled distribution; a `pantr-metis` wheel is on the later list (QB8).
Why the deflation is a change, not a port: today's `v0 = arange(m) - (m-1)/2` is exactly orthogonal to the constant
vector, so in exact arithmetic the Krylov space never contains the `lambda_1 = 0` eigenvector, and the code takes
`argsort(values)[1]` as the Fiedler vector. Measured on main's scipy 1.17.1: for a 31 x 31
grid graph (961 vertices) `eigsh` returned `[0.01026, 0.09172]` against dense `[~0, 0.01026, 0.01026]`, so the vector
used belongs to `0.0917`, not to the Fiedler value; for a 3000-vertex path ARPACK did not converge and the code fell back
to a dense 3000 x 3000 `eigh`. The verification pass reproduced the same trap in Spectra (a start vector with no component along
an eigenvector misses it while reporting `Successful`). Deflating the known null vector makes the requested eigenpair
the Fiedler pair by construction. This is a defect met in passing on main (B candidate, section 15 item 11); its fix
lands in stage 6 as its own commit with a regression test that fails on main first (the Rayleigh quotient of the
chosen vector equals `lambda_2` of the dense solve within a derived tolerance, on the 961-vertex grid).

`test_eigsh_failure_falls_back_to_dense` monkeypatches the private scipy `eigsh`; it is rebound to a private testing
hook (`_pantr._testing.spectral_partition(..., lanczos_max_iter=1)`) that forces the fallback.

## 5. MPI

### 5.1 Shape

```cpp
namespace pantr::mpi {
template <class C> concept Communicator = requires(const C c, std::span<const double> d, std::span<const std::int64_t> i) {
  { c.rank() } -> std::convertible_to<int>;  { c.size() } -> std::convertible_to<int>;
  { c.allgatherv(d) } -> std::same_as<Gathered<double>>;          // flat data + counts, rank order
  { c.allgatherv(i) } -> std::same_as<Gathered<std::int64_t>>;
  { c.allreduce_sum(d) } -> std::same_as<std::vector<double>>;
};
class AnyCommunicator;   // type-erased over comm_capsule.h's C vtable; MPI-free
// algorithms/*.hpp: create_distributed_space, interpolate (staged + callable), fit, l2_project, quasi_interpolate[_thb],
//                   partition_from_owner (from_dolfinx's assembly) ... all templated on Communicator, none includes <mpi.h>
class MpiCommunicator;   // mpi_communicator.hpp, pantr::mpi target, includes <mpi.h>; plus overloads taking MPI_Comm
}
```

Kernels take arrays and scalars; only `MpiCommunicator` sees MPI types; core headers never include `mpi.h` (AC4's
check: a CI step compiles every core header with an `mpi.h` on the include path that is `#error`, plus a grep).

### 5.2 How the Python suite stays the oracle (a conflict, flagged)

There is a genuine tension: seven test files drive `pantr.mpi` with duck-typed fakes (`_FakeComm` with preset
`allgather_results`, `_AllreduceHub` summing real tensors), which a C++ algorithm cannot call if it only accepts
`MPI_Comm`. Resolution: the algorithms are compiled **once, in `_pantr`**, instantiated with `AnyCommunicator`. Two
producers of an `AnyCommunicator`:

- `PyCommunicator` (in `_pantr`, no MPI): calls the Python object's `rank`, `size`, `allgather`, `allreduce` with the
  **same payloads the Python implementation sends today** (an ndarray; a tuple `(owned_global_dofs, owned_coeffs)` for
  QI; `allreduce` of an ndarray with the default op). The fake-communicator tests therefore keep their preset payloads
  and exercise the C++ algorithms, single-process, deterministically.
- `_pantr_mpi`: wraps an mpi4py communicator into the same C vtable backed by `MpiCommunicator` (real `MPI_Allgatherv`,
  `MPI_Allreduce`).

`pantr.mpi` picks the producer by type: an `mpi4py.MPI.Comm` goes through `_pantr_mpi` (required, with the family
check), anything else through `PyCommunicator`. The duck-typed path is today's documented contract (`comm: Any`),
so it is kept, not test-only.

Residual cost, stated: where a fake presets a payload in a format the C++ algorithm no longer produces, that test
changes, and the change is listed under AC1 as a replaced test. The design keeps payload formats to make that set
empty; it cannot prove it is empty before stage 8.

Why not the alternatives: (a) keep orchestration in Python and call C++ only for local compute: contradicts R0's
"distributed logic in C++" and leaves a C++ user with no distributed API; (b) MPI-only C++ with fakes replaced by
`mpiexec` tests: loses the fast single-process multi-rank simulations the suite relies on, and turns "the suite is the
oracle" into "a different suite"; (c) `_pantr_mpi` containing the algorithms: duplicates every numeric it inlines into
a module that is built once per MPI family and never per ISA, and would pass C++ objects across extension modules.

### 5.3 Rest of the layer

- `DistributedSpace` reads `rank/size` only, builds `LocalSpace` (core); `from_dolfinx` stays duck-typed Python for the
  dolfinx mesh access and calls C++ for the owner assembly and the collective.
- Thread policy: the Python default (1 pool thread per rank unless explicitly configured) is kept;
  `configure_threads(threads_per_rank)` stays, `limit_blas` goes (QB2). C++ `pantr::mpi` sets no thread count.
- The `_l2.py` docstring says each rank evaluates `func` only on its owned cells; the code evaluates the global lattice
  and masks. The port keeps the behaviour and corrects the docstring (evaluating only owned cells is on
  the later list as a speed-up).
- Collectives raise on a payload of unexpected size; a rank-local exception before a collective still deadlocks (as
  today; documented Note kept).

### 5.4 Payloads of the fake communicators (U2, read in stage 0)

Every fake communicator the suite defines, and what the production code sends through it.
Production touches a communicator only through `comm.size`, `comm.rank` (Python attributes or
properties, read as `int(...)`), `comm.allgather(obj)` and `comm.allreduce(arr)` with no `op`
argument (mpi4py's default, sum); nothing calls `bcast`, `Barrier`, `Get_rank` or `Get_size`.

| Test file (fake) | How it fakes | Production call and payload | Fake's result |
|---|---|---|---|
| `tests/test_mpi_collocation.py:45` `_FakeComm` | preset list returned whatever is sent; one rank: `[data]` | `mpi/_collocation.py:342` and `:457` `allgather` of float64 `(m,)` or `(m, r)`, C-contiguous; `m` may be 0 | preset `list` of float64 `(m_r,)` or `(m_r, 2)` blocks |
| `tests/test_mpi_qi.py:39` `_FakeComm` | as above | `mpi/_qi.py:123` `allgather` of `(int64 (k,), float64 (k, r))`; an empty rank sends `(0,)` and `(0, 0)` | preset `list` of such tuples |
| `tests/test_mpi_thb_qi.py:39` `_FakeComm` | as above | `mpi/_thb_qi.py:134` `allgather` of `(int64 (k,), float64 (k,) or (k, r))`; an empty rank sends `(0,)` and `(0,)` even for a vector problem | preset `list` of such tuples |
| `tests/test_mpi_l2.py:43,89` `_AllreduceHub`, `_FakeComm` | a hub summing the ranks' tensors in two passes (collect, then frozen); one rank: returns `data` | `mpi/_l2.py:265` `allreduce` of a float32 **or** float64 tensor `(*num_basis, n_comp)`; every rank sends the full shape | elementwise sum, dtype kept; the fake's `allreduce` takes one positional argument |
| `tests/test_distributed_space.py:40` `_FakeComm` | `rank` and `size` only | `mpi/_distributed_space.py:95,107`, `mpi/_create.py:85` read `size`/`rank`; `ds.comm is comm` is asserted (`:111`) | none |
| `tests/test_mpi_thread_policy.py:31` `_FakeComm` | one rank; `rank`, `size` class attributes; `allgather` returns `[sendobj]` | `mpi/_from_dolfinx.py:92` `allgather` of int64 `(k,)` | `[that array]` |
| `tests/test_from_dolfinx.py:20` `_FakeComm` | `size` only (no `rank`); `allgather` puts the sent array at the caller's slot and preset arrays elsewhere | `mpi/_from_dolfinx.py:92` `allgather` of int64 `(k,)`, `(0,)` for an empty rank | `list` of int64 1-D arrays of length `size` |
| `tests/test_mpi.py` | no fake communicator (it fakes the `mpi4py` import) | n/a | n/a |

What this means for `PyCommunicator` (DR-4), to settle when stage 8 starts:

- The tuple payloads need two gathers (int64, then float64) and the column count `r` carried
  separately, since a flat float64 gather loses it; the consumers skip empty blocks, so the `(0,)`
  versus `(0, 0)` difference of an empty rank is not observable.
- `allreduce_sum` must accept float32 as well as float64: the L2 projection sends a float32 tensor
  for a float32 space, and a hub that sums in float32 is what its test compares against.
- `rank` and `size` are read as attributes, never called; one fake has no `rank` at all, which works
  today because `from_dolfinx` reads only `size`. An adapter that requires `rank` there would break it.
- The object passed in must stay the object returned (`DistributedSpace.comm is comm`).
- The preset fakes ignore what the calling rank sends, so in their multi-rank simulations only the
  single-rank runs check the payload production builds.

## 6. Dense linear algebra on Eigen

numpy's decision rules are reproduced as named constants citing numpy's source (V5), because several
outcomes the suite asserts are discrete.

| Site | Today | Eigen | Discrete hazard and handling |
|---|---|---|---|
| `_bspline_interpolate._solve_1d` (square) | `solve` (gesv), SVD fallback on `LinAlgError` | `PartialPivLU`; fallback when a U pivot is exactly 0 | numpy raises when gesv reports `info > 0` (an exactly zero U pivot; on OpenBLAS also some NaN inputs) and never on conditioning; Eigen never raises, so the test is explicit: a U pivot that is exactly 0 **or NaN** takes the fallback. Whether a near-singular matrix yields an exact zero pivot is BLAS-dependent already, so no test may assert that branch on such a matrix. |
| `_solve_1d` rectangular, Bezier fits | truncated SVD, `100*eps*s_max` | `JacobiSVD` (small) / `BDCSVD` | rank at the threshold can flip with last bits; the threshold is relative and derived; tests at the threshold are the hazard (U3). |
| `change_basis` (4 sites) | `solve(G, C)`, `solve(forward, I)` | `PartialPivLU` | none beyond conditioning, which the degree limits already bound. |
| `_bspline_knot_insertion.to_periodic` | `qr`, `solve(R, Q.T@b)` | `HouseholderQR` + triangular solve | residual check raises on gross violation: threshold unchanged. |
| products (`multiply`, 6 sites), THB prolongation | `lstsq(rcond=None)` (gelsd) | `BDCSVD`, cutoff `eps * max(m, n) * s_max` (`_linalg.py:2530`, gelsd's `S(i) <= RCOND*S(1)`) | rank decision at the cutoff; min-norm solution identical in exact arithmetic. |
| THB restriction | `pinv` | `BDCSVD`, cutoff `1e-15 * s_max` (numpy 2.4 default, `_linalg.py:2240`; not `max(m,n)*eps`, which is `lstsq`'s) | as above. |
| QI local blocks | `inv`, singular -> `ValueError` | `PartialPivLU`, exact-zero-pivot -> `value_error` | same message. |
| `locate` health check | `svd` of the Jacobian | `JacobiSVD` (tiny) | singular-value ratio test unchanged. |
| spectral partition | `eigh` / `eigsh` | `SelfAdjointEigenSolver` / Spectra on the Laplacian **deflated against the constant vector** (4.2) | **eigenvector sign** canonicalised as today; **near-degenerate lambda_2** makes the Fiedler vector arbitrary within its eigenspace, so a partition can differ from ARPACK's. `test_partition_graph.py` asserts properties (balance, separation, validity) and determinism within a process, not exact memberships (read by its test names and its `assert_array_equal` sites, not every body), so the hazard is bounded to those tests' properties. |
| `quad`: `leggauss`, Lobatto roots, `chebpts1` | numpy polynomial | Newton on the three-term recurrence (lesson 1), closed form | nodes move in the last bits; tests use tolerances. |

Eigen's kernels replace OpenBLAS/Accelerate, so the "local BLAS differs from CI BLAS" trap (CLAUDE.md) disappears; in
its place, Eigen's SIMD paths differ per ISA rung, which is why the wheel job runs the suite once per reachable rung.

## 7. Other numerical replacements

- Lagrange tabulation (decided before this design): barycentric weights by the product formula accumulated as (mantissa, exponent) pairs
  (`frexp`) instead of scipy's random permutation, so overflow and underflow are avoided deterministically; the
  `16*eps` node snap kept; tests keep their tolerances.
- tanh-sinh step: `lambertw` replaced by Halley's iteration on the principal branch over the argument range the rule
  uses, with a both-terms convergence test (*a convergence test has both terms*) and a loud cap.
- Exact rational assembly of degree-reduction operators: Boost.Multiprecision `cpp_rational` (standalone), cached per
  `(degree, decrement)` (2.2). In-house bigint was the alternative: rejected because exact Gaussian elimination is
  where a hand-written bignum gets its rare carry bug, and no test of the float result would see it.
- `_bincoeff`: exact `uint64` recurrence with `__builtin_mul_overflow` (MSVC `_umul128`) so overflow is detected rather
  than wrapping; the public envelope `_BINCOEFF_MAX_N = 61` and its guard are unchanged (signed overflow is undefined
  behaviour in C++, so reproducing Numba's wrap is neither possible nor wanted).
- `math.comb` big-int sites (`_bezier_product`, `_bezier_eval`, `_bezier_compose`, `_bspline_eval`): the exact `uint64`
  recurrence within the envelope, `cpp_int` beyond it where today's code goes beyond it.

## 8. Test architecture (tests:design)

### 8.1 Layer ownership

- **Python suite: owns numerical correctness of the public API** (the oracle, R0). Every changed assertion is one of
  four AC1 categories: derived tolerance, independent oracle replacing a mirror, replaced Numba-specific test, API change
  in the changelog. Each stage's commit group lists its changed assertions by category.
- **C++ suite (doctest): owns what Python cannot observe**: construction validation and exception types of the C++ API,
  `[[nodiscard]]` statuses, mdspan/`out` overloads and extents checks, the pool (concurrent callers, nesting, exception
  propagation, thread-count invariance), every lazy block and process table under TSan, the provider contract suite
  (4.1), `Communicator` algorithms with an in-process fake and with real MPI (`ctest` under `mpiexec -n 2,3`), plus a
  small known-answer battery per module from `tests/oracles` (so a C++-only consumer has regression coverage without
  duplicating the Python suite).
- **The seam no unit test covers**: the binding. Covered by the Python suite running end to end with no mocks, plus a
  Python test per extension boundary (`_pantr` <-> `pantr_berf`, `_pantr` <-> `_pantr_mpi`) passing real arrays.

### 8.2 Framework and legs

doctest (MIT). CI (every push of the port branch; nothing reduced):

1. lint: ruff, ruff format, import-lint, `clang-format --dry-run`.
2. type-check: mypy strict on 3.12, 3.13, 3.14 against the committed `_pantr.pyi`.
3. tests: build `_pantr` once (abi3, cp312) and run Python suite + doctests + coverage on 3.12, 3.13, 3.14 (the abi3
   check is real: one binary, three interpreters); stub-diff check.
4. docs: Sphinx `-W`.
5. cpp: GCC 14 and Clang (Release, `-Werror`, `PANTR_NATIVE_ARCH=OFF`) at baseline, `x86-64-v3`, and forced scalar.
6. cpp-asan-ubsan: Clang, `-fno-sanitize-recover=undefined` (lesson 2: otherwise UBSan reports and exits 0).
7. cpp-tsan: the pool, lazy blocks, process tables, concurrent `parallel_for` from many threads.
8. install-consumer: `cmake --install` to a fresh prefix, build `cpp/tests/consumer` with `find_package(pantr COMPONENTS
   mpi)` and no Python; plus the `mpi.h` poison-header compile of every core header (AC4).
9. mpi: real MPI, `mpiexec -n 2` and `-n 3`, Python `tests/mpi/` and C++ MPI tests; `_pantr_mpi` built from source
   against the runner's MPI.
10. sweep: `_pantr` built with ASan+UBSan, `-D_GLIBCXX_ASSERTIONS`, `PANTR_ASSERT` on; runs `tools/adversarial_sweep`
    and the slow-marked tests (replaces `NUMBA_BOUNDSCHECK=1`).
11. wheels: cibuildwheel (manylinux_2_28 x86_64 and aarch64, macOS arm64, Windows amd64, R8) for `pantr`,
    `pantr-mpi`, `pantr-berf`; content check (no BeRF-FR header in `pantr`, AC7); the installed-wheel suite run once per
    reachable rung (`PANTR_ISA=generic` and the best rung; `v4` only where the runner has it), AC8.
12. no-mpi: `import pantr`, `import pantr.mpi` and the fake-communicator tests with no MPI and no mpi4py installed;
    mismatched-family test (AC10).
13. berf: `-DPANTR_WITH_BERF=ON` C++ build + contract suite, `pantr-berf` built and the Python provider-switch test
    both ways (AC7). Needs BeRF-FR source in CI (R7).

`make test` keeps `pytest -n auto` for CI runners behind a variable and local runs pass `-n 8`.

### 8.3 Shared oracles and the mirrors M1 to M7

`tests/oracles/` (Python, built once, reused by every module and by the C++ battery through generated data files):
exact-rational Cox-de Boor and de Casteljau (`fractions.Fraction`), exact Bernstein Gram and elevation matrices,
polynomials with encoded roots, closed-form quadrature moments, metamorphic helpers (shift/scale/rotate carrying the
tolerance). Generated C++ data files are produced from these, never from pantr.

| Mirror | Replacement oracle (the old test is kept, re-pointed or demoted to a consistency check, never deleted) |
|---|---|
| M1 `_eval_periodic_correct` (3 files) | exact-rational Cox-de Boor on the unwrapped knot vector (definition, not the NURBS-book triangle), plus periodicity `f(t + T) = f(t)` and partition of unity |
| M2 cached vs uncached change_basis | exact-rational change-of-basis matrices at low degree; the cache test becomes a C++ test of table identity and immutability |
| M3 Bernstein round trip with shared forward operator | forward samples from exact-rational de Casteljau |
| M4 formula pins | kept as deliberate **specification pins** (the formula is the spec), labelled as such, read through `pantr._testing` |
| M5 dense Oslo / O(n) blossom references | exact-rational knot insertion (Boehm, from the definition) and blossom by the polar-form definition; plus "insertion preserves the curve" at random parameters |
| M6 batch vs single roots | kept as an API-consistency property, plus encoded-truth polynomials for batch |
| M7 serial vs parallel twins | thread-count invariance: the same input under `num_threads(1)` and `num_threads(k)`, sized across the grain boundary, within a derived tolerance |

### 8.4 Numba-specific tests

| Test | Becomes |
|---|---|
| `test_jit_warmup_barrier.py` (+ the two monkeypatch sentinels) | a subprocess test that starts N Python threads calling parallel entry points as the first thing in a fresh process (the abort pattern), asserting success; and the C++ TSan test of concurrent callers. The barrier itself is deleted with Numba. |
| `test_parallel.py` | get/set/context tests kept; max = pool capacity (from `os.sched_getaffinity` on Linux, `PANTR_NUM_THREADS` override tested); `limit_blas` tests removed with the API (QB2, changelog). |
| serial/parallel parity (`test_bspline_space_1D.py:2176-2585`) | M7 above. |
| `test_adversarial_sweep.py` (`NUMBA_BOUNDSCHECK`) | same driver, run in the sweep leg against the sanitized build. |
| `test_bincoeff_envelope.py::test_wraps_one_past_the_declared_bound` | "one past the bound the guarded entry raises `capacity_error`" in Python, and a C++ test that the checked recurrence detects overflow at `n = 62`. |
| `conftest.py` thread save/restore | `pantr.get_num_threads` save/restore; `_reset_policy_for_testing` kept. |
| `test_review_regressions.py:46` (`_JIT_DISABLED`) | branch removed. |
| `test_mpi_thread_policy.py::...limit_blas_warns_without_threadpoolctl` | removed with the API (changelog). |

## 9. Performance

### 9.1 Operations benchmarked (AC6), each owned by the stage that ports it

Per operation, sizes `tiny` (one point or one object, the per-call overhead regime), `medium` (1e3 items) and `large`
(1e6 points, or the largest a stage's test fixtures build), at pantr threads 1 and 8:

- S1: `tabulate_bernstein/cardinal_bspline/legendre/lagrange[_derivatives]_1d`, ND `tabulate_*`, `get_gauss_legendre_1d`
  (n = 8, 64), `get_tanh_sinh_1d`, `compute_*_1d` change-of-basis (first call and repeated), `tolerance.get_*`.
- S2: `AABB` ops, `AffineTransform` apply; `Bezier` construct, `evaluate`, `evaluate_derivatives`, `split`, `elevate/reduce_degree`, `minimize_degree`,
  `interpolate_bezier`, `fit_bezier`, `find_roots` single (degree 3, 12) and batch (1e4 polynomials), `find_monotone_root`.
- S4: `BsplineSpace1D` construct, `tabulate_basis` (1D, 2D, 3D), `Bspline` construct (now a copy), `evaluate`
  (1D/2D/3D), `evaluate_derivatives`, `insert_knots`, `remove_knots`, `elevate_degree`, `to_beziers`, `multiply`,
  `to_periodic`, `find_roots`.
- S5: `interpolate_bspline`, `fit_bspline` (tensor, scattered), `l2_project_bspline`, `quasi_interpolate_bspline`,
  spanwise extraction operator/apply.
- S6: `TensorProductGrid.locate_many`, `HierarchicalGrid.refine`, BVH `query_aabb`, `Bspline.locate`, THB
  `tabulate_basis`, `prolongation_to_sparse`, `quasi_interpolate_thb_spline`, `MultiLevelExtraction`, `coupling_graph`,
  `partition_graph`, `partition_grid`.
- S7: `create_coons_surface`, `create_revolution`, `join`, `detect_interfaces`.
- S8: each distributed function on `n = 1` (fake communicator) and `mpiexec -n 2`.

### 9.2 Harness and statistic

In-repo `benchmarks/` (no new dependency): each case is a named callable plus setup; a runner measures with
`time.perf_counter_ns` in auto-ranged loops, writes JSON with commit, machine, env, ISA rung, thread settings.
Protocol: K = 5 independent processes, each R = 30 repeats after warm-up (Numba compiled and cached before timing);
per process the median; **branch statistic = median of the 5 process medians; spread = max minus min of main's 5
process medians**. AC6 holds per operation when `branch <= main + spread_main`. Between-process spread is the honest
noise floor (*establish the baseline spread first*: environment changes reorder variants).

### 9.3 Baseline capture

Stage 0 captures main 11ab436 on the server: `taskset` to a fixed set of 8 cores, `OMP_NUM_THREADS = OPENBLAS_NUM_THREADS
= MKL_NUM_THREADS = 1`, `NUMBA_NUM_THREADS` and `pantr.set_num_threads` at 1 and 8, Numba cache warm, `cpupower`
state recorded if readable. The same settings apply to every branch measurement; the results live under
`benchmarks/results/` beside the command that produced them (never in a durable doc,
*no measured number in anything durable*). Effective compile flags of the measured `.so` are checked first (the `-Os`
trap).

### 9.4 Stage-0 abi3 measurement (R6)

Three builds of a stage-0 `_pantr` exposing a handful of S1 functions: abi3 `NB_STATIC`, version-specific `NB_STATIC`,
and main's Numba path, on the `tiny` cases (one point, one tiny array, a property read). If abi3's per-call cost puts
any `tiny` case over main's statistic, stop and return to the maintainer with split mode (R6).

## 10. Stages

Each stage: C++ headers + bindings + wrappers + tests for its modules, Numba kernels of those modules deleted, its
benchmarks green against the baseline, the downstream re-census where a symbol is reshaped, full CI green, one
commit group. Size estimates: kind (new 3 / larger extension 2 / small edit 1), size
(about one per primary file read whole or written, more if heavy), difficulty left to the builder after this design
(design 3 / limited design 2 / mechanical 1); the plan cuts the build items from them.

| Stage | Component | Kind | Size | Diff. | Reason |
|---|---|---|---|---|---|
| S0 foundation | C0.1 CMake skeleton, options, probes, flags, deps, install/export/config, consumer project | 3 | 8 | 2 | pattern fixed here and in lessons; install/export is new but standard |
| | C0.2 core headers: config, errors+translator, format (`py_repr`), ndarray, mdspan alias, lazy, simd, exact_rational shim | 3 | 7 | 2 | shapes pinned in section 2; `py_repr` exactness is fiddly |
| | C0.3 pool, `parallel_for`, affinity, thread-local count, work model, Python `_parallel` on C++ | 3 | 4 | 3 | concurrency semantics (caller participates, nesting, exceptions, determinism) have real forks at the code level |
| | C0.4 packaging: pyproject, bindings CMake, multi-ISA, `_pantr_cpu`, `_isa.py`, `_ext.py` guards, make targets, stubs | 3 | 9 | 2 | copied scheme, new loader and staleness guards |
| | C0.5 CI rewrite (13 legs, 8.2) | 2 | 4 | 2 | heavy YAML, every leg named here |
| | C0.6 benchmark harness, baseline capture, abi3 measurement | 3 | 4 | 2 | protocol fixed in section 9 |
| S1 quad, tolerance, basis, change_basis | C1.1 C++ quad (GL Newton, GLL, Chebyshev, trapezoid, tanh-sinh + Lambert W), PointsLattice, QuadratureRule | 3 | 6 | 2 | algorithms known; Lambert W pinned in 7 |
| | C1.2 C++ tolerance, basis (Bernstein, cardinal, Legendre, Lagrange barycentric, derivatives, ND tabulate) | 3 | 9 | 2 | 9 jit kernels + Layer 2; Lagrange pinned in 7 |
| | C1.3 change_basis (Eigen, degree limits, bounded tables) | 3 | 4 | 2 | |
| | C1.4 bindings + wrappers + tests (basis, quad, change_basis, tolerance, bincoeff, Bernstein underflow, warmup and parallel replacements, M2) | 2 | 10 | 2 | many test files, message porting |
| S2 bezier (+ geometry, transform) | C2.1 AABB, AffineTransform (C++, wrappers, `__eq__`/`__hash__`) | 3 | 4 | 1 | small value types |
| | C2.2 Bezier value type + eval/deriv/split/slice/restrict/collapse/compose/product/elevate | 3 | 14 | 2 | 7 parallel kernels, handle swap, overloads |
| | C2.3 degree reduction/minimisation (exact rational), interpolate/fit (SVD) | 3 | 5 | 2 | |
| | C2.4 root seam + Builtin provider (Yuksel, clipping with statuses, dedup, batch, monotone), Python switch | 3 | 8 | 2 | contract pinned in 4.1; dedup pair-class table to write |
| | C2.5 tests: root finding (M6), degree reduction, interpolate (M3), changelog entries | 2 | 8 | 2 | |
| S3 berf provider | C3.1 `pantr::berf::Provider` adapter, exhaustive mapping, contract suite, `pantr-berf` packaging and CI leg | 3 | 6 | 3 | the claim mapping is a judgement against BeRF-FR's documented claims; BeRF-FR API read in depth |
| S4 bspline core | C4.1 BsplineSpace1D/BsplineSpace, knots, basis core, factories, Greville | 3 | 12 | 2 | 8+7 jit kernels, eager derived state |
| | C4.2 Bspline value type, eval/derivative, knot insertion/removal, degree, blossom, restrict/slice/split, to_beziers, conversions, periodic (QR) | 3 | 18 | 2 | the bulk of the library; M1, M5 |
| | C4.3 B-spline roots (Morken-Reimers, status -> warning), products (lstsq) | 3 | 6 | 2 | |
| | C4.4 tests and private-symbol re-pointing for bspline core (largest test files) | 2 | 12 | 2 | `test_bspline_space_1D.py` alone is heavy |
| S5 fitting and extraction | C5.1 interpolate/fit/L2/QI with the staged callable shape | 3 | 8 | 2 | staged API pinned in 3.6 |
| | C5.2 extraction: spanwise, kernels (one mode-wise contraction primitive, lesson 1), helpers, `_bspline_extraction` | 3 | 8 | 2 | 24 jit kernels collapse to one primitive |
| S6 grid, locate, THB, partition | C6.1 TensorProductGrid, HierarchicalGrid (version, handle swap), Grid name, BVH, tags, cell_quadrature, overlay, Partition, partition_grid | 3 | 14 | 2 | grid jit kernels; CRTP defaults |
| | C6.2 Bspline.locate (Newton + BVH) | 3 | 3 | 2 | heavy single file |
| | C6.3 THBSplineSpace/THBSpline/THB QI/multilevel extraction/restriction, local space, halo, dof_owner, coupling graph | 3 | 12 | 2 | 2380-line space; stale-grid check in the wrapper |
| | C6.4 spectral partition (Eigen/Spectra), METIS C++ provider, pymetis path | 3 | 4 | 2 | |
| | C6.5 tests (grid private attributes, THB, partition, locate) | 2 | 10 | 2 | |
| S7 cad, multipatch | C7.1 cad (8 files), multipatch (5 files) in C++ with wrappers and tests | 3 | 14 | 2 | pure Python today; validation messages to port |
| S8 mpi | C8.1 `pantr/mpi` algorithms, `AnyCommunicator`, capsule ABI, `MpiCommunicator`, C++ MPI tests | 3 | 8 | 3 | payload compatibility with the fakes and the FUNNELED discipline are code-level forks |
| | C8.2 `PyCommunicator`, `_pantr_mpi`, `pantr-mpi` packaging, family check, wrappers, thread policy, tests | 3 | 9 | 2 | dolfinx scheme confirmed at build time |
| S9 retirement | C9.1 remove numba/threadpoolctl/`_numba_compat`/warmup/NUMBA env; AC5 import check; final full benchmark run; changelog and docs (install, C++ guide); design notes | 2 | 6 | 1 | mechanical once S1 to S8 are green |

Ordering: S0, S1, S2, then S3 and S4 may run in either order (S3 needs only S2), S5 after S4, S6 after S5 (THB uses
extraction and locate uses Bezier patches), S7 after S6 (cad/multipatch use bspline and transform), S8 after S6, S9
last. Tests touched per stage are the files importing that stage's modules.

## 11. Third-party dependencies and licences (AC12)

| Dependency | Use | Licence | Ships in |
|---|---|---|---|
| Eigen 5.0.1 (or 3.4.1) | dense LA | MPL-2.0 (`EIGEN_MPL2_ONLY` defined) | compiled into extensions; headers bundled by `cmake --install` |
| Spectra 1.2.0 | sparse Lanczos | MPL-2.0 | as Eigen |
| Kokkos mdspan 0.6.0 | `mdspan` in C++20 | Apache-2.0 WITH LLVM-exception | as Eigen |
| Boost.Multiprecision 4.1+ standalone (`BOOST_MP_STANDALONE`, with its bundled Boost.Config) | exact rationals | BSL-1.0 | as Eigen |
| nanobind (exact pin) | binding | BSD-3-Clause | statically in each extension |
| scikit-build-core | build backend | Apache-2.0 | build only |
| doctest 2.5.3 | C++ tests | MIT | tests only |
| numpy | runtime | BSD-3-Clause | dependency |
| scipy | `csr_array` at the boundary | BSD-3-Clause | dependency |
| mpi4py | `pantr-mpi` | BSD-3-Clause | dependency of `pantr-mpi` |
| MPI library (MPICH / Open MPI / Intel MPI) | `pantr-mpi` runtime | permissive for MPICH and Open MPI, proprietary-redistributable for Intel MPI (recalled, not checked; confirm in stage 8; irrelevant to pantr's licence since nothing is bundled) | user-installed, not bundled |
| pymetis | `pantr[metis]` | MIT (to confirm in stage 6) | optional dependency |
| METIS | `pantr::metis` (C++) | Apache-2.0 | user-provided |
| BeRF-FR | `pantr-berf` only | its author's hybrid licence (planned) | `pantr-berf` wheel only |

None of these makes the `pantr` wheel or `pantr::pantr` anything but MIT-distributable. MPL-2.0 is file-level copyleft:
the bundled headers are shipped unmodified with their licence files (a NOTICE listing all in the wheel).

## 12. Lessons reused, re-decided, and the later list

Reused (lesson 1 rows): ownership classes; immutability (now with the R5 handle swap); grouped lazy block with
double-checked locking; C++ owns domain types; dtype as output format where it is one today; pickle via constructor
arguments; one mode-wise extraction primitive with runtime dimension; feature probes; never fast-math, contraction on
the exported target; SIMD policy; batched callbacks; scalar-generic kernels; CRTP grid; own BVH; Gauss-Legendre by
Newton; the parity-rule vocabulary for tolerances in tests; every trap of lesson 2.

Re-decided: kernel seam and dual-backend parity machinery (gone with one backend); "no process-global cache" narrowed to
integer-keyed bounded tables (2.2); validation (lessons 3.8): C++ validates in every build, not `assert`; errors map to
an explicit hierarchy, not "type checks in the wrapper only"; `format_repr` is needed again because the suite matches
messages (it was "dual"); dense LA is Eigen (lessons 3.1, closed by R0) with numpy's decision rules reproduced; threading
model (3.4) designed here; Python-defined grids cannot exist (R1); `mdspan` chosen by a CMake option, not a header test.

Later list: smoothing/penalised fits, node layout as a type, THB L2 per level (R2; the staged fitting API and value
types admit them without breaking); AD `Dual` (R3); requested minimum claim and claim-returning Python API on the root
seam; curve-curve intersection; subset extraction API (lesson 1, extraction API row); `pantr-metis` wheel; split-mode
nanobind and free-threaded builds; cgroup-quota-aware pool sizing; L2 distributed evaluation on owned cells only;
explicit SIMD kernels where auto-vectorisation fails; an optional precompiled instantiation library; musllinux wheels;
Windows `pantr-mpi` (Intel MPI); speed-ups beyond parity found during the port.

## 13. Decisions that deserve a note in `design/` (expensive to reverse)

One short note each, written in the stage that lands it: (1) value types with shared immutable representation and the
Python handle swap; (2) thin Python wrappers over private nanobind classes, and `pantr._testing` as the only private
test surface; (3) the root-finding seam contract (claims, completeness, capabilities, wire values pinned, dedup
pair-class table); (4) the communicator concept, the capsule C ABI and the `PyCommunicator` path; (5) packaging: three
distributions, abi3 `NB_STATIC`, multi-ISA ladder, headers in the wheel, arrays-only between extensions; (6) the
threading model (thread-local parallelism, fixed chunking, no cross-chunk reductions); (7) the error model (hierarchy,
verbatim messages, status-to-warning); (8) `mdspan` identity by CMake option.

## 14. Acceptance criteria: how each is made achievable and checked

| AC | Made achievable by | Checked by |
|---|---|---|
| AC1 suite, doctests, real MPI | wrappers keep the API; verbatim messages; `PyCommunicator`; 4-category ledger of changed assertions | CI legs 3, 9; the per-stage ledger in each commit group |
| AC2 docs `-W` | wrappers keep docstrings; stubs | leg 4 |
| AC3 C++ suite, GCC+Clang, sanitizers, TSan on pool and caches | 8.1 ownership; every lazy block has a race test | legs 5, 6, 7 |
| AC4 `find_package` consumer, no `mpi.h` in core | 1.5 install/export with bundled deps; 5.1 header split | leg 8 (consumer build and run, poison `mpi.h`, grep) |
| AC5 no numba/threadpoolctl; scipy only at the boundary | S9 removal; 3.5 | an import check in leg 1: AST scan of `src/pantr` for `numba`, `threadpoolctl`, and `scipy` outside the allow-listed wrapper modules, plus `pip show` of the installed deps |
| AC6 no public op slower | section 9 | benchmark runs per stage on the server with the 9.2 statistic, results committed beside the command |
| AC7 BeRF-FR separation and switch | 1.6, 4.1 | leg 11 content listing; leg 13 switch test both ways; default C++ build has no `berf` include path (CMake test) |
| AC8 multi-ISA ladder | 1.8 | leg 11: suite per reachable rung; `_isa` unit tests with a monkeypatched probe for every rung, including `PANTR_ISA=<missing>` raising |
| AC9 changelog | each stage appends its entries (R5 list, `Grid` subclassing, `limit_blas`, root-provider API, pickling now explicit) | review at each stage gate; a test that every name the changelog lists as removed is absent |
| AC10 separate `_pantr_mpi`, import without MPI, mismatch reported | 1.7 | leg 12 |
| AC11 METIS optional | 4.2 | leg 5 with `PANTR_WITH_METIS=ON` on one job; Python metis tests with pymetis installed |
| AC12 licences | section 11 | the table; a CI step listing `FetchContent` declarations against it |
| AC13 stage gates | section 10 | each stage's last commit is green on the full branch CI before the next begins |

## 15. Flagged practices (R0, rulings, current code)

1. **Silent caps in Bezier clipping** (`bezier/_clipping_core.py:116, 152-294`): a full stack or root buffer drops work
   with no signal (*exhaustion raises never returns*). The port removes the Numba-imposed stack cap and reports the
   depth cap (2.7, QB7).
2. **Literal reading of "Kokkos only where `<mdspan>` is missing"** would create an ODR hazard; replaced by a CMake
   option (2.6).
3. **Pickling is implicit today** and would vanish silently with a nanobind handle; made explicit (3.2).
4. **Thread count semantics**: Numba's count is thread-local; a naive pool would make `pantr.num_threads()` in one
   Python thread affect another. Kept thread-local (2.8).
5. **The root-finding default tolerance's frame**: a dimensionless preset used as an absolute bracket
   width and coefficient cutoff. Kept, frame stated, routed to the tolerance gate.
6. **`Makefile` `test` uses `pytest -n auto`**, which sizes itself from the machine rather than the CPUs allowed (fixed in P0a).
7. **`_l2.py` docstring contradicts the code**: docstring corrected in S8.
8. **Error messages as a tested contract** (852 `match=`): legitimate, but it makes every message part of the API; the
   C++ messages become the single source and Python never re-raises with a reworded text.
9. **`_CLIP_COEFF_RANGE_LIMIT` duplicated "keep in sync"** in two files because Numba kernels cannot import constants:
   one C++ constant.
10. **`eigsh` falls back to dense `eigh` silently**: the fallback stays (it computes the same quantity)
    but is recorded in `PartitionInfo`.
11. **Spectral partition above 512 vertices can use a non-Fiedler vector** (`bspline/_partition_graph.py:275-278`,
    measured, 4.2): `v0` orthogonal to the null vector. Fixed in stage 6 with its own regression test (B1 in the plan).

## 16. Questions, as settled

The two calls that needed the maintainer were ruled: BeRF-FR's source reaches CI by FetchContent
from its public repository at a pinned SHA, the stage that needs it moving late (R7); wheels are
built for Linux x86_64/aarch64, macOS arm64 and Windows amd64, `pantr-mpi` for Linux and macOS (R8).
The defaults this design proposed became the plan's QB4 (nanobind 3.1.x pinned exactly, falling
back to 2.14.x if stage 0 fails), QB5 (Eigen 5.0.1 if Spectra passes on it, else 3.4.1;
`EIGEN_MPL2_ONLY` guards only direct inclusion of the non-MPL-2.0 files, so it is a tripwire, not a
guarantee), QB6 (`pantr-mpi` installing `pantr_mpi`, `pantr-berf` installing `pantr_berf`, no file
of one distribution inside another's package), QB7 (the Bezier clipping depth cap warns through
`RuntimeWarning`; a test that reaches it then fails under `filterwarnings = error`, which exposes a
silent truncation of today, and that test is fixed rather than the warning suppressed), QB8
(`pantr[metis]` keeps pymetis; the C++ METIS provider is for C++ users), QB9 (Boost.Multiprecision
standalone for exact rationals), QB10 (the root-provider switch as new public API) and QB11 (one
pool thread per rank under `pantr.mpi`).

## 17. Unverified items

- **U1** Cross-ISA layout independence of public types holds once written: the cheapest check is a C++ test that
  `static_assert`s `sizeof`/`alignof` of every public type against a table, compiled in the baseline and v4 legs.
- **U2** (settled in stage 0, section 5.4) `PyCommunicator` can reproduce every payload the fake-communicator tests
  preset, provided it gathers tuples in two parts, sums float32 as well as float64, and reads `rank`/`size` as
  attributes.
- **U3** No test asserts a rank decision exactly at an SVD threshold: grep for tests constructing rank-deficient fits
  at `100*eps` and run them on two ISA rungs in stage 2.
- **U4** The Python wrapper frame plus abi3 call fits under main's `tiny` timings: the stage-0 measurement (9.4).
- **U5** dolfinx's family check details have not moved since #4652: re-read the merged code at stage 8 start.
- **U6** Spectra with ARPACK's default `ncv` reproduces `eigsh`'s convergence on the suite's graphs: run
  `test_partition_graph.py` at stage 6 with the fallback counter asserted zero on the non-fallback tests.
- **U7** pymetis licence (stated as MIT, recalled, not checked): read its PyPI page in stage 6.
- **U8** Kokkos mdspan's namespace-selection macro name (README did not show it): read `mdspan/include/experimental/__p0009_bits/config.hpp`
  in stage 0 before writing `core/mdspan.hpp`.
- **U9** `import_mpi4py()` under the stable ABI: compiled with `-DMPI4PY_LIMITED_API`, never run; run it in stage 8's
  first build of `_pantr_mpi`.

## 17b. Type design analysis (four axes, judged separately; categories, not scores)

| Type | Encapsulation | Invariant expression | Invariant usefulness | Invariant enforcement | Weakest, and what raises it |
|---|---|---|---|---|---|
| `Bspline<T>` and the other domain values | strong: representation behind `shared_ptr<const Rep>`, no mutator | medium: immutability is in the type (`const Rep`), shape/dtype agreement only in the constructor | high: kills the aliasing and stale-cache bugs the map observed (`reverse()` view, caches not reset on writes) | strong: validating constructor is the only door; numeric views are `const` | expression: knot/coefficient count agreement is runtime; a typed `Extents` would raise it but costs every call site, not worth it |
| `RootSet<T>` / `Claim` / `Completeness` | open struct (fields public) | strong: claim and completeness are closed enums with pinned values | high: makes a dropped region or an uncertified root visible | medium: a consumer can read `roots` and ignore `completeness`; `[[nodiscard]]` stops only discarding the whole result | enforcement: a Python-side warning covers Python users; for C++ an accessor that throws on incomplete unless `allow_incomplete()` was asked would raise it, at the cost of an extra call for every consumer; not chosen, revisit if a C++ consumer misses a gap |
| `Pool` / `parallel_for` | strong: workers private, one entry point | medium: determinism (chunking independent of thread count) is a documented property, not typed | high: the concurrency abort Numba had is a real, reachable failure | strong for safety (caller participation, exception capture); determinism enforced by the M7 test only | expression: a `Reduction` type that fixes chunk order would make the no-cross-chunk-reduction rule structural if a reduction is ever added |
| `AnyCommunicator` + capsule C ABI | strong: vtable opaque | medium: versioned struct, version checked at conversion | high: the only way fakes and MPI share one algorithm | medium: a wrong payload size is caught at the collective, not at compile time | enforcement: `Gathered<T>` validates counts sum to data length at construction |
| Python wrappers (`Bspline`, ...) | medium: `_h` is private by convention only | low: Python cannot express immutability of `_h`; `in_place` swaps are by design | high: R5 needs exactly this hole | medium: every mutation path is a method that rebuilds `_h` through a validating C++ constructor | encapsulation: `__slots__` plus a `__setattr__` that refuses `_h` outside the class's own methods would raise it; lesson 1 called that wrapper memo "dual"; not chosen unless a test needs it |

Design rules applied, each with how strongly the case for it is made here: *an abstraction earns its
place by a contract*, argued (the root and partition seams each have a second implementor, BeRF-FR
and METIS, before they are built; `Communicator` has three: fake, MPI, in-process C++ test); *hide
the decision most likely to change*, argued (the three changes most likely in a year: a new root
provider touches `providers/` only; a new MPI family touches `_pantr_mpi` only; a new fitting
variant touches the staged fitting API and its wrapper only); *make a speed mutation unobservable*,
proved as to mechanism (the handle swap mutates only the wrapper, never a value another holder
sees); *a closed set needs exhaustive dispatch*, a house rule (the BeRF-FR claim mapping); *convert
a foreign interface at a boundary*, proved as to mechanism (mpi4py communicator, pymetis output,
scipy.sparse are converted at one seam each); *cross a boundary once*, measured elsewhere with the
transfer argued (callables evaluated once per batch; arrays, not objects, between extensions).

## 18. Verified claims (V1 to V12, checked independently before approval)

| Claim the design rests on | Verdict | What it changed here |
|---|---|---|
| V1 no `<mdspan>` in g++ 14.4 (C++20 or C++23) or clang 18 on libstdc++ | VERIFIED (compiled) | 2.6 strengthened: Kokkos is the practical default |
| V2 `register_exception_translator`; defaults invalid_argument/domain/length/range -> ValueError, out_of_range -> IndexError, overflow -> OverflowError, bad_alloc -> MemoryError, rest -> RuntimeError; user translators run first | VERIFIED (2.14.0 and v3.1.0 source, runtime probe) | 2.5 unchanged; the explicit translator is still registered so no mapping rests on defaults |
| V3 zero-copy `nb::ndarray` owned by `nb::capsule` | VERIFIED (docs, header, runtime: `owndata=False`, deleter after last view) | 2.4 |
| V4 mpi4py 4.1.2 `mpi4py.h`: `import_mpi4py()`, `PyMPIComm_Get` | VERIFIED (compiled, incl. `MPI4PY_LIMITED_API`; not run) | 1.7; U9 |
| V5 numpy 2.4.6: `pinv` cutoff `1e-15*s_max`; `lstsq` `eps*max(m,n)`; `solve` raises on gesv `info != 0` | VERIFIED; (c) qualified: NaN can give `info > 0`, near-singular never raises, exact-zero pivots are BLAS-dependent | section 6 rows rewritten |
| V6 Spectra 1.2.0: header-only, MPL-2.0, `SmallestAlge`, `init(resid)`; works with Eigen 3.4.0 and 5.0.1 in one probe; a start vector missing an eigen-direction misses that eigenvalue with `Successful` | VERIFIED with caveat | 4.2 deflation; QB2 prefers Eigen 5.0.1 |
| V7 Boost.Multiprecision standalone since 4.1.0 (Boost 1.79), BSL-1.0, needs Boost.Config | VERIFIED with qualifier | 11 |
| V8 `EIGEN_MPL2_ONLY` errors only on direct inclusion of non-MPL2 files; system Eigen is 3.4.0 | VERIFIED with qualifier | QB5 wording |
| V9 Kokkos mdspan Apache-2.0 WITH LLVM-exception; `operator()` default, macros select | VERIFIED; namespace macro UNVERIFIED | U8 |
| V10 dolfinx #4652 merged (be8352e), wheel-only family check | VERIFIED with qualifiers (family-level, injected at wheel build) | 1.7 |
| V11 doctest MIT, 2.5.3 | VERIFIED | 11 |
| V12 scikit-build-core 1.0.3: `editable.mode = "inplace"`, `install.strip`, `wheel.py-api` | VERIFIED | 1.9 (inplace mode not used; the `make ext` route stays) |

The `eigsh` measurement of section 4.2 was made by the design's author and is not among the
independently checked claims.
