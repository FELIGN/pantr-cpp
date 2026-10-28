# Third-party software in pantr's C++ build

pantr itself is MIT-licensed (`LICENSE`). Its C++ build fetches the dependencies below, each pinned
to a full commit SHA in `cmake/PantrDependencies.cmake`; update this file together with any pin
there. All are header-only. `cmake --install` copies the headers of those marked "installed",
unmodified and with their licence files, under `include/pantr/third_party/<name>/`.

| Dependency | Version | Commit | Upstream | Licence (SPDX) | Use | Distribution |
|---|---|---|---|---|---|---|
| Eigen | 5.0.1 | `bc3b39870ecb690a623a3f49149a358b95c5781d` | https://gitlab.com/libeigen/eigen | `MPL-2.0` (see below) | dense linear algebra | installed |
| Spectra | 1.2.0 | `6841bcbacaa0f0a8446210314e682057a084be4e` | https://github.com/yixuan/spectra | `MPL-2.0` | sparse symmetric eigensolver | installed |
| Kokkos mdspan | 0.6.0 | `9ceface91483775a6c74d06ebf717bbb2768452f` | https://github.com/kokkos/mdspan | `Apache-2.0 WITH LLVM-exception` | `mdspan` for C++20 | installed unless `PANTR_USE_STD_MDSPAN` |
| Boost.Multiprecision | Boost 1.90.0 | `529dfac199191a7eb8a5eb7f47256eff6d0db993` | https://github.com/boostorg/multiprecision | `BSL-1.0` | exact rationals (standalone mode) | installed |
| Boost.Config | Boost 1.90.0 | `a7d5a9b05d70c9cfea980dc3539ca3d3461411b3` | https://github.com/boostorg/config | `BSL-1.0` | required by standalone Boost.Multiprecision | installed |
| doctest | 2.5.3 | `2d0a9359a60c51affe2a9bebb1be1dca47868151` | https://github.com/doctest/doctest | `MIT` | C++ test framework | tests only, never installed |

## Notes

- **Eigen** is primarily MPL-2.0. Some of its files carry code under other MPL-2.0-compatible
  licences (BSD, Apache-2.0, MINPACK); its `COPYING.*` files, installed beside the headers, list
  them. pantr defines `EIGEN_MPL2_ONLY` on `pantr::pantr`, which makes Eigen refuse direct inclusion
  of its non-MPL-2.0 modules; that is a tripwire against an accidental include, not a guarantee
  about every file.
- **MPL-2.0** is file-level copyleft: the Eigen and Spectra headers are shipped unmodified, with
  their licence text, which keeps pantr distributable under MIT.
- **Boost.Multiprecision and Boost.Config** share one installed `boost/` tree and the Boost
  Software License text from Boost.Multiprecision. A consumer that also uses its own Boost should
  make sure only one `boost/config.hpp` is reachable.
- **Kokkos mdspan 0.6.0** declares its types in `std::experimental`.

## Not bundled

Python runtime dependencies (numpy, scipy, and the others in `pyproject.toml`) are installed by the
Python package manager and are not part of the C++ build. An MPI library, when `pantr::mpi` is
used, is the user's own and is never bundled.
