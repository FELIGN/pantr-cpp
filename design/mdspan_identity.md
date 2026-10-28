# mdspan identity

Decision record for which type `pantr::mdspan` names (`include/pantr/core/mdspan.hpp`). Expensive
to reverse because every N-dimensional array argument of the C++ API is spelled with it.
Architecture: `design/port_design.md` §2.6; plan directive Di5.

## The decision

- `pantr::mdspan` (with `extents`, `dextents`, the layouts, and the row-major alias `mdview<T, R>`)
  is `std::mdspan` when `PANTR_USE_STD_MDSPAN` is defined, otherwise the Kokkos reference
  implementation (mdspan 0.6.0) that pantr bundles.
- **The choice is made once, at configure time.** A probe compiles a `std::mdspan` snippet at the
  C++ standard the build uses and sets `PANTR_USE_STD_MDSPAN` as a compile definition on the
  exported `pantr::pantr` target, so every translation unit of a consumer program sees the same
  type. A builder may set the CMake variable `PANTR_USE_STD_MDSPAN` explicitly; turning it ON where
  the probe fails is a configure error.
- **Why not a header test.** `#if __has_include(<mdspan>)` in the header would give `pantr::mdspan`
  two meanings in a program that mixes C++20 and C++23 translation units: an ODR violation in every
  signature that mentions it, with no diagnostic. R0 asks for the Kokkos implementation "only where
  `<mdspan>` is missing"; the configure-time probe keeps that intent with one type per program.
- **Subscripts are fixed too.** The Kokkos implementation enables `operator()` or a multi-index
  `operator[]` depending on the language version of each translation unit. pantr fixes
  `MDSPAN_USE_PAREN_OPERATOR=1` and `MDSPAN_USE_BRACKET_OPERATOR=0` on the target, so the class is
  the same in C++20 and C++23 units, and pantr code indexes with `operator()`.

## As built (stage 0)

- No toolchain the port targets today has `<mdspan>`: GCC 14's libstdc++ has none in C++20 or C++23
  mode, nor Clang 18 on libstdc++. The bundled Kokkos version is therefore the practical default.
- Kokkos mdspan 0.6.0 declares its types in `std::experimental` and has no macro to choose that
  namespace (the plan's U8 asked; the macros appeared only in later, untagged versions). A consumer
  that bundles a different Kokkos mdspan version in the same program would collide on those names.
- `std::mdspan` has no `operator()`, which pantr code uses. Until that is settled (how pantr's code
  indexes in that configuration, with the first kernel that takes an `mdspan`, stage 1), the probe
  only reports whether `<mdspan>` exists, and setting `PANTR_USE_STD_MDSPAN=ON` is a configure
  error rather than a build that cannot compile.
