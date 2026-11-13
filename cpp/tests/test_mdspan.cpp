// pantr::mdspan: the alias names one implementation, indexed with operator(), row-major.
#include <doctest/doctest.h>

#include <pantr/core/mdspan.hpp>

#include <array>
#include <cstddef>
#include <type_traits>

TEST_CASE("mdspan: the alias and the configured implementation agree") {
#if defined(PANTR_USE_STD_MDSPAN)
  static_assert(std::is_same_v<pantr::mdspan<double, pantr::dextents<std::size_t, 2>>,
                               std::mdspan<double, std::dextents<std::size_t, 2>>>);
#else
  static_assert(std::is_same_v<
                pantr::mdspan<double, pantr::dextents<std::size_t, 2>>,
                std::experimental::mdspan<double, std::experimental::dextents<std::size_t, 2>>>);
  // The subscript macros are fixed on the target, not left to the language version.
  static_assert(MDSPAN_USE_PAREN_OPERATOR == 1 && MDSPAN_USE_BRACKET_OPERATOR == 0);
#endif
  CHECK(true);
}

TEST_CASE("mdview: row-major, operator() indexing, const views") {
  std::array<double, 6> data{0, 1, 2, 3, 4, 5};
  pantr::mdview<double, 2> m(data.data(), 2, 3);
  CHECK(m(0, 0) == 0.0);
  CHECK(m(0, 2) == 2.0);
  CHECK(m(1, 0) == 3.0);
  m(1, 2) = 50.0;
  CHECK(data[5] == 50.0);
  pantr::mdview<const double, 2> c = m;
  CHECK(c(1, 2) == 50.0);
  static_assert(std::is_const_v<std::remove_reference_t<decltype(c(0, 0))>>);
}
