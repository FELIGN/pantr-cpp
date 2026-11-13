/// @file mdspan.hpp
/// @brief `pantr::mdspan` and its companions: one multidimensional view type per program.
///
/// C++20 has no `std::mdspan`. pantr uses the standard one when the build defines
/// `PANTR_USE_STD_MDSPAN`, else the Kokkos reference implementation bundled with pantr. The choice
/// is a compile definition on the exported `pantr::pantr` target, decided once when pantr is
/// configured, so every translation unit of a program sees the same type; a `__has_include` test
/// here would give two types in a program mixing C++20 and C++23 translation units.
///
/// The bundled Kokkos version declares its types in `std::experimental`. The target also fixes
/// its subscript macros (`MDSPAN_USE_PAREN_OPERATOR=1`, `MDSPAN_USE_BRACKET_OPERATOR=0`), which
/// otherwise follow the language version of each translation unit; pantr code indexes with
/// `operator()`. Design note: design/mdspan_identity.md.
#pragma once

#include "pantr/config.hpp"

#include <cstddef>

#if defined(PANTR_USE_STD_MDSPAN)
#include <mdspan>
#else
#include <experimental/mdspan>
#endif

namespace pantr {

#if defined(PANTR_USE_STD_MDSPAN)
namespace mdspan_ns = ::std;
#else
namespace mdspan_ns = ::std::experimental;
#endif

/// The multidimensional view type (`std::mdspan` or the Kokkos reference implementation).
using mdspan_ns::mdspan;
/// Extents with static and dynamic components.
using mdspan_ns::extents;
/// All-dynamic extents of a given rank.
using mdspan_ns::dextents;
/// Row-major (C order) layout, the layout of every pantr array.
using mdspan_ns::layout_right;
/// Column-major (Fortran order) layout.
using mdspan_ns::layout_left;
/// Arbitrary-stride layout.
using mdspan_ns::layout_stride;

/// Row-major view with all-dynamic extents of rank `Rank` and `std::size_t` indices.
template <class T, std::size_t Rank>
using mdview = mdspan<T, dextents<std::size_t, Rank>, layout_right>;

}  // namespace pantr
