.PHONY: help ext stubs stubs-check cpp test doctest coverage clean install ruff-lint ruff-format ruff-format-check type-check import-lint pre-pull-request docs

help:
	@echo "Commands:"
	@echo "  ext       : build the compiled extension modules and install them into src/pantr."
	@echo "  stubs     : regenerate the committed type stubs of the extension modules."
	@echo "  stubs-check: fail when the committed stubs differ from the current build's."
	@echo "  cpp       : build and run the C++ test suite (gcc-release preset)."
	@echo "  test      : run the test suite."
	@echo "  doctest   : run the docstring examples in src/pantr."
	@echo "  coverage  : generate a coverage report."
	@echo "  clean     : remove build artifacts."
	@echo "  install   : install project with dev extras."
	@echo "  ruff-lint : run Ruff linter."
	@echo "  ruff-format : check Ruff formatting changing files."
	@echo "  ruff-format-check : check Ruff formatting without changing files."
	@echo "  type-check: run mypy static type checker."
	@echo "  import-lint: check import boundaries (core must not import pantr.mpi)."
	@echo "  docs      : build the documentation."
	@echo "  pre-pull-request: build the extension, check its stubs, run the C++ suite, then lint, format, format check, type check, import lint, tests, coverage, and docs."

# Build the compiled extension and install it into src/pantr, where pytest.ini's `pythonpath = src`
# finds it: no editable install, so every worktree runs its own build (design/packaging.md). The
# defaults build what a wheel ships (stable ABI, every ISA rung), plus the test-only modules.
# Modules of a previous build are removed first: Python prefers a version-specific suffix over
# `.abi3`, so a leftover module would shadow the new one. The installed modules are then touched:
# install keeps the build's timestamps, and a source edit that changed no object (a comment, a
# header nothing includes) must still clear the test session's staleness check (tests/conftest.py).
EXT_BUILD_DIR ?= build/dev
EXT_STABLE_ABI ?= ON
EXT_MULTI_ISA ?= ON
EXT_CMAKE_ARGS ?=
ext:
	cmake -S . -B $(EXT_BUILD_DIR) -G Ninja -DCMAKE_BUILD_TYPE=Release \
	  -DPython_EXECUTABLE="$$(python -c 'import sys; print(sys.executable)')" \
	  -DFETCHCONTENT_BASE_DIR="$(CURDIR)/build/_deps" \
	  -DPANTR_BUILD_PYTHON=ON -DPANTR_BUILD_TESTS=OFF -DPANTR_INSTALL=OFF \
	  -DPANTR_NATIVE_ARCH=OFF -DPANTR_STABLE_ABI=$(EXT_STABLE_ABI) \
	  -DPANTR_MULTI_ISA=$(EXT_MULTI_ISA) -DPANTR_BUILD_PYTHON_TEST_MODULES=ON $(EXT_CMAKE_ARGS)
	cmake --build $(EXT_BUILD_DIR)
	find src/pantr -maxdepth 1 -name '_pantr*.so' -delete
	cmake --install $(EXT_BUILD_DIR) --component python --prefix "$(CURDIR)/src"
	cmake --install $(EXT_BUILD_DIR) --component testing --prefix "$(CURDIR)/src"
	find src/pantr -maxdepth 1 -name '_pantr*.so' -exec touch {} +

# Regenerate the stubs mypy reads (committed; the CI tests job fails when they differ from the
# build's). Needs a current `make ext`. `_pantr` is a module with a submodule, so its stub is a
# package directory (src/pantr/_pantr/__init__.pyi); the compiled module still wins at import, as
# a directory without __init__.py is only a namespace candidate. PANTR_ISA=generic: importing pantr
# must load the same rung stubgen reads, or the rung's classes register twice.
STUBGEN = PANTR_ISA=generic PYTHONPATH="$(CURDIR)/src" python -W error -m nanobind.stubgen -q -P \
  --exclude-values -p tools/stubgen_patterns.txt
stubs:
	$(STUBGEN) -r -m pantr._pantr -O src/pantr
	$(STUBGEN) -m pantr._pantr_cpu -o src/pantr/_pantr_cpu.pyi
	ruff check --fix-only -q src/pantr/_pantr src/pantr/_pantr_cpu.pyi
	ruff format -q src/pantr/_pantr src/pantr/_pantr_cpu.pyi

# Regenerate the stubs and fail if that changed the committed ones (a binding changed without
# `make stubs`).
STUB_PATHS = src/pantr/_pantr src/pantr/_pantr_cpu.pyi
stubs-check: stubs
	git diff --exit-code -- $(STUB_PATHS)
	test -z "$$(git ls-files --others --exclude-standard -- $(STUB_PATHS))"

# The C++ suite under the release preset (the sanitizer and Clang presets: CLAUDE.md).
cpp:
	cmake --preset gcc-release
	cmake --build --preset gcc-release
	ctest --preset gcc-release

# Run the test suite with Numba JIT enabled. PYTEST_WORKERS defaults to 8, not auto:
# pytest-xdist sizes `auto` from os.cpu_count(), which ignores the process affinity mask,
# so on a shared many-core machine it starts far more workers than the CPUs allowed. CI
# runners are sized for their job, so CI passes PYTEST_WORKERS=auto explicitly.
PYTEST_WORKERS ?= 8
test:
	pytest -n $(PYTEST_WORKERS)

# Run the docstring examples shipped in the package sources. Kept out of `test`
# because `testpaths = tests` in pytest.ini deliberately excludes src/, and a plain
# `pytest` run should stay the fast inner loop. No coverage and no xdist: the whole
# set runs in under a second, and worker startup would dominate.
#
# NUMBA_DISABLE_JIT=1 is deliberate. With JIT off, `prange` is `range` and no Numba
# threading layer is ever entered, so this run cannot race the background warmup thread
# that pantr/__init__.py starts: the concurrent-compilation abort class is structurally
# absent here, not merely improbable. It is also 28x faster (0.9s against 24s cold),
# which matters now that `pre-pull-request` depends on it.
#
# Historical note, so nobody "restores" the JIT here: the JIT-enabled run this replaced
# measured 0 aborts in 44 runs (24 cold-cache, 20 warm), but was safe only because
# pytest happens to collect basis/_basis_1D.py early and its examples reach a
# `wait_for_jit_warmup()` call site before any unguarded kernel. Reordering the modules
# or dropping those examples would have removed that protection silently.
#
# What this gives up is the compiled path, which is `test`'s job over the whole suite.
# This target checks that the documentation matches the code, and the values it asserts
# go through np.allclose or .tolist(), neither of which depends on JIT-vs-interpreter
# rounding.
doctest:
	NUMBA_DISABLE_JIT=1 pytest --doctest-modules src/pantr

# Generate an XML coverage report with Numba JIT disabled
coverage:
	COVERAGE_FILE=/tmp/.coverage NUMBA_DISABLE_JIT=1 pytest -m "not slow" --cov=src/pantr --cov-report=term-missing --cov-report=xml

# Remove build artifacts
clean:
	rm -rf .pytest_cache .coverage coverage.xml htmlcov/

# Install project with development dependencies
install:
	python -m pip install --upgrade pip
	pip install -e ".[dev]"

# Ruff linting
ruff-lint:
	ruff check .

# Ruff formatting check (changes performed)
ruff-format:
	ruff format .

# Ruff formatting check (no changes written)
ruff-format-check:
	ruff format --check .

# Static type checking
type-check:
	mypy --config-file mypy.ini src tests

# Import boundary checks: serial core must not import pantr.mpi.
# PYTHONPATH pins the analysis to this checkout's source: import-linter resolves `pantr`
# through sys.path, so without it the contract is checked against whatever the editable
# install points at, which in a git worktree is a different tree entirely. The path is
# absolute because a relative one resolves against whatever working directory the command
# is given (`conda run` may give another checkout's), and `tests._tree_env` first proves
# that `pantr` resolves to this tree.
import-lint:
	PYTHONPATH="$(CURDIR)/src" python -m tests._tree_env
	PYTHONPATH="$(CURDIR)/src" lint-imports

# Build documentation
docs:
	$(MAKE) -C docs html SPHINXOPTS="$(SPHINXOPTS)"

# Aggregate target to run all checks before creating a pull request
pre-pull-request: ext stubs-check cpp ruff-lint ruff-format ruff-format-check type-check import-lint test doctest coverage docs
