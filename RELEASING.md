# Releasing PaNTr

Maintainer runbook for cutting a release. Publishing is automated by
[`.github/workflows/release.yaml`](.github/workflows/release.yaml): pushing a
`v*` tag builds the distributions of **pantr** and **pantr-mpi** and publishes
them to TestPyPI and then PyPI via
[PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC — no
API tokens are stored anywhere), then cuts a GitHub Release.

The distributions are built by
[`.github/workflows/wheels.yaml`](.github/workflows/wheels.yaml), the same
reusable workflow CI runs as leg 11 on every push, so what ships is what CI
built and tested:

- pantr: one cp312-abi3 wheel per platform (Linux x86_64 and aarch64, macOS
  arm64, Windows amd64), each built by cibuildwheel and tested against the suite
  under its rungs, and the sdist;
- pantr-mpi: one wheel per platform (Linux x86_64 and aarch64 against MPICH,
  whose `libmpi.so.12` is the MPICH ABI; macOS arm64 against Open MPI; no
  Windows wheel), with `PANTR_MPI_FAMILY` set explicitly and the MPI library not
  bundled, and the sdist, which CI also builds into a wheel against the pantr
  wheel (the route of a user with no matching wheel). pantr-mpi requires
  `mpi4py>=4` (its stable-ABI C API).

The version is single-sourced from `__version__` in `src/pantr/__init__.py`
(scikit-build-core reads it with a regex, for pantr and for pantr-mpi). **The
git tag must be `v<version>`** — the pipeline fails the release if any built
wheel's version does not match the tag.

## One-time setup

These are dashboard actions, done once for the project. They cannot be committed
to the repository.

### PyPI and TestPyPI trusted publishers

On both [pypi.org](https://pypi.org/manage/account/publishing/) **and**
[test.pypi.org](https://test.pypi.org/manage/account/publishing/) (enable 2FA on
each account first), add a *pending publisher*:

| Field             | Value          |
| ----------------- | -------------- |
| PyPI Project Name | `pantr`, and a second publisher for `pantr-mpi` |
| Owner             | `FELIGN`       |
| Repository name   | `pantr`        |
| Workflow name     | `release.yaml` |
| Environment name  | `pypi` on pypi.org, `testpypi` on test.pypi.org |

The first successful publish creates the project; the pending publisher then
becomes a regular trusted publisher.

### GitHub environments

In *Settings → Environments*, create two environments named `pypi` and
`testpypi` (the names must match the pending publishers above). Add yourself as a
*required reviewer* on each so a tag push pauses for a one-click approval before
anything is uploaded.

### Read the Docs

Confirm the GitHub webhook is connected (RTD → *Admin → Integrations*; a recent
delivery should show `200`). RTD reads `.readthedocs.yaml` from the repo, so no
other configuration is needed. Optionally add an *Automation Rule* matching the
`v*` tag pattern to **activate** the new version and **set it as default** — with
that rule in place the per-release RTD steps below become automatic.

## Cutting a release

1. **Land all changes on `main`** with CI green.
2. **Bump the version**, in every place that states it (CMake and the test
   suite refuse a disagreement, so a missed one fails the checks):
   - `__version__` in `src/pantr/__init__.py`, the single source;
   - `PANTR_VERSION_MAJOR`, `_MINOR` and `_PATCH` in `include/pantr/version.hpp`
     (the root `CMakeLists.txt` refuses to configure when they differ);
   - the `pantr-mpi==<version>` pin of the `mpi` extra in `pyproject.toml`, and
     the `pantr==<version>` build requirement in `bindings/mpi/pyproject.toml`
     (`tests/test_mpi.py::test_mpi_extra_pins_pantr_mpi_at_this_version` checks
     both). pantr-mpi's sdist builds against an installed pantr of that exact
     version; in the release run, the pantr wheels of the same run satisfy it.
3. **Update the changelog**: add the new `## <version> (<date>)` section to
   `docs/changelog.md` (Keep a Changelog style — `### Added` / `### Changed` /
   …). This exact section becomes the GitHub Release notes.
4. Open a PR with the bump + changelog, run the checks, and merge it.
5. *(Optional) Rehearse.* Trigger the workflow manually from *Actions → Release →
   Run workflow*. This builds and publishes to **TestPyPI only** — production
   PyPI is never touched on a manual run. Verify the install:

   ```bash
   pip install --index-url https://test.pypi.org/simple/ \
     --extra-index-url https://pypi.org/simple/ "pantr==<version>"
   ```

6. **Tag and push** from `main`:

   ```bash
   git checkout main && git pull
   git tag v<version>
   git push origin v<version>
   ```

7. **Approve** the `testpypi` and then `pypi` environments when the run pauses
   (*Actions* tab). The pipeline then publishes to PyPI and cuts the GitHub
   Release.
8. **On Read the Docs** (unless an Automation Rule handles it): *Versions* →
   activate `v<version>`; *Admin → Settings* → set it as the default version.

## Verify

- PyPI project page shows the new version, and `pip install "pantr==<version>"`
  works from a clean environment.
- The GitHub Release exists with the changelog section as its notes and every
  wheel and both sdists attached; `pip install "pantr[mpi]==<version>"` installs
  pantr-mpi on Linux and macOS.
- `https://pantr.readthedocs.io` serves the new version's docs.

## How the pipeline is wired

`release.yaml` runs five jobs:

- **wheels** — calls `wheels.yaml`: the cibuildwheel wheels and the sdists of
  both distributions, each wheel tested.
- **collect** — splits the artifacts into `dist-pantr` and `dist-pantr-mpi`,
  runs `twine check --strict`, and on a tag asserts every wheel's version equals
  the tag.
- **testpypi** — publishes both to TestPyPI (`skip-existing`, so re-runs are
  safe). Runs on both a tag push and a manual dispatch.
- **pypi** — publishes both to PyPI. Gated to tag pushes only.
- **github-release** — creates the GitHub Release with notes extracted from the
  matching section of `docs/changelog.md`.

Permissions are least-privilege: the publish jobs request only the OIDC
`id-token` they need, and the third-party publish action is pinned to a commit
SHA.
