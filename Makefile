.PHONY: help ext stubs stubs-check cpp test doctest coverage clean install ruff-lint ruff-format ruff-format-check type-check import-lint dependency-check pre-pull-request docs

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
	@echo "  dependency-check: no numba or threadpoolctl; scipy only in its boundary modules."
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
	find src -maxdepth 2 -path 'src/pantr_mpi/*' -name '_pantr_mpi*.so' -delete
	cmake --install $(EXT_BUILD_DIR) --component mpi --prefix "$(CURDIR)/src"
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

# Run the test suite. PYTEST_WORKERS defaults to 8, not auto:
# pytest-xdist sizes `auto` from os.cpu_count(), which ignores the process affinity mask,
# so on a shared many-core machine it starts far more workers than the CPUs allowed. CI
# runners are sized for their job, so CI passes PYTEST_WORKERS=auto explicitly.
PYTEST_WORKERS ?= 8
# The extension loads the highest ISA rung the CPU supports, so a second pass pins the
# generic rung (no FMA): a result that depends on the rung otherwise shows only in CI's wheels.
test:
	pytest -n $(PYTEST_WORKERS)
	PANTR_ISA=generic pytest -n $(PYTEST_WORKERS)

# Run the docstring examples shipped in the package sources. Kept out of `test`
# because `testpaths = tests` in pytest.ini deliberately excludes src/, and a plain
# `pytest` run should stay the fast inner loop. No coverage and no xdist: the whole
# set runs in about a second, and worker startup would dominate.
doctest:
	pytest --doctest-modules src/pantr

# Generate an XML coverage report
coverage:
	COVERAGE_FILE=/tmp/.coverage pytest -m "not slow" --cov=src/pantr --cov-report=term-missing --cov-report=xml

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

# Retired dependencies (numba, threadpoolctl) stay out of every manifest and module, and scipy is
# imported only where a public function returns a scipy object (tools/check_dependencies.py).
dependency-check:
	python tools/check_dependencies.py

# Build documentation
docs:
	$(MAKE) -C docs html SPHINXOPTS="$(SPHINXOPTS)"

# Aggregate target to run all checks before creating a pull request
pre-pull-request: ext stubs-check cpp ruff-lint ruff-format ruff-format-check type-check import-lint dependency-check test doctest coverage docs
