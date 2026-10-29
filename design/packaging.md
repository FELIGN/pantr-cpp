# Packaging

Decision record for how pantr is built and shipped as Python packages. Expensive to reverse
because every wheel, every downstream build of a pantr extension module and the development loop
of every contributor rest on it. Architecture: `design/port_design.md` §1.6 to §1.9.

## The decision

- **Three distributions.** `pantr` (the core and its extension), `pantr-mpi` (`_pantr_mpi`, the MPI
  glue, built against one MPI family per platform) and `pantr-berf` (the BeRF-FR root-finding
  provider, under its own licence). `pantr[mpi]` and `pantr[berf]` pull the other two at the same
  version. The `pantr` wheel and the default C++ build never contain a BeRF-FR file.
- **nanobind, statically, for the stable ABI.** Every module is built `NB_STATIC STABLE_ABI
  NOMINSIZE NOSTRIP NB_SUPPRESS_WARNINGS`, so one `cp312-abi3` wheel per platform serves CPython
  3.12 and later. nanobind is pinned exactly in `pyproject.toml`, and the CMake build refuses any
  other version: modules that exchange nanobind objects must share its ABI, so `pantr-mpi` and
  `pantr-berf` build against the same pin.
- **Multi-ISA rungs.** The extension is compiled once per rung under different names (x86-64:
  `_pantr`, `_pantr_v3`, `_pantr_v4`; arm64: `_pantr` forced scalar and `_pantr_neon`).
  `pantr/_isa.py` imports the best rung the CPU and OS support, asking `_pantr_cpu` (CPUID and
  XGETBV, built at the baseline with no pantr header, failing closed). `PANTR_ISA` names exactly
  one rung and never falls through; the loaded module reports its rung as `isa`. Two rungs never
  load in one process (their classes would register twice).
- **One pool per process across modules.** `_pantr` owns the thread pool and the per-thread
  settings; another pantr module attaches its table at import (`design/threading.md`).
- **Arrays only between extensions.** Extension modules exchange arrays, scalars and capsules of
  versioned C structs, never each other's C++ types, so their binaries need not agree on anything
  beyond those layouts and the nanobind pin.
- **Headers in the wheel**, so that `pantr-mpi` and `pantr-berf` can be built from their sdists
  against an installed `pantr` (§1.5). Not built yet: it lands with the first distribution that
  needs it (the MPI stage), together with a CMake package whose bundled dependencies the wheel
  does not carry.
- **No editable install.** `make ext` builds into `build/dev` and installs the modules into
  `src/pantr`, where `pytest.ini`'s `pythonpath = src` finds them, so every worktree tests its own
  build. Two guards keep a stale module from passing as somebody else's regression: the module's
  `interface_version` must equal `pantr._ext.EXPECTED_INTERFACE`, and the test session refuses a
  module older than any source under `include/`, `bindings/` or `providers/*/include`.
- **The installed wheel is tested as such.** `tools/wheel_test.py` runs the suite with
  `PANTR_TEST_INSTALLED=1` and an empty `pythonpath`, once under the default rung and once under
  `generic`, and the session refuses a `pantr` that resolves to the checkout.

## Why

- abi3 with `NB_STATIC` rather than nanobind's split mode: one wheel per platform, no separate
  backend distribution to keep in step. The cost is a per-call overhead in the limited API, which
  the plan required measuring before committing to it: on the `tiny` cases the stable-ABI build
  stayed within main's own spread of the version-specific build, so the split-mode alternative was
  not triggered (`benchmarks/results/`, the r6 runs).
- `NOMINSIZE NOSTRIP` and `install.strip = false`: nanobind appends `-Os` after `-O3` (the last flag
  wins) and links release modules stripped, and scikit-build-core strips again at install. Each is
  silent; CI checks the build's commands and the binaries.
- No editable install: scikit-build-core's editable mode puts a finder ahead of `sys.path`, so a
  worktree silently imports another checkout's package.

## Alternatives not taken

- Version-specific wheels: one wheel per Python version and platform, for a per-call difference
  measured as within noise.
- GNU unique symbols to share the pool: ELF and GCC only.
