// py_repr against CPython's own repr. Oracle: the expected strings were produced by CPython's
// repr(float) on the exact doubles written here in hexadecimal, an implementation independent of
// pantr's (CPython formats with its own dtoa).
#include <doctest/doctest.h>

#include <pantr/core/format.hpp>

#include <array>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace {

struct Case {
  double value;
  const char* repr;
};

// Positional/exponential boundary cases (1e-4 vs 1e-5, 1e15 vs 1e16), signed zero, subnormals,
// the extremes, and 17-significant-digit values.
constexpr std::array<Case, 29> kCases{{
    {0x0.0p+0, "0.0"},
    {-0x0.0p+0, "-0.0"},
    {0x1.0000000000000p+0, "1.0"},
    {-0x1.0000000000000p+0, "-1.0"},
    {0x1.999999999999ap-4, "0.1"},
    {0x1.5555555555555p-2, "0.3333333333333333"},
    {0x1.9000000000000p+6, "100.0"},
    {0x1.a36e2eb1c432dp-14, "0.0001"},
    {0x1.4f8b588e368f1p-17, "1e-05"},
    {0x1.f75104d551d69p-14, "0.00012"},
    {0x1.f75104d551d69p-17, "1.5e-05"},
    {0x1.c6bf526340000p+49, "1000000000000000.0"},
    {0x1.1c37937e08000p+53, "1e+16"},
    {0x1.1c37937e07fffp+53, "9999999999999998.0"},
    {0x1.aa535d3d0c000p+53, "1.5e+16"},
    {0x1.b69b4ba630f35p+56, "1.2345678901234568e+17"},
    {0x1.0f0cf064dd592p+73, "1e+22"},
    {0x1.0c6f7a0b5ed8dp-22, "2.5e-07"},
    {0x0.0000000000001p-1022, "5e-324"},
    {0x1.fffffffffffffp+1023, "1.7976931348623157e+308"},
    {0x1.0000000000000p-1022, "2.2250738585072014e-308"},
    {-0x1.edd2f1a9fbe77p+6, "-123.456"},
    {0x1.8000000000000p+1, "3.0"},
    {0x1.34a0000000000p+10, "1234.5"},
    {0x1.0000000000000p-1, "0.5"},
    {0x1.249ad2594c37dp+332, "1e+100"},
    {0x1.bff2ee48e0530p-333, "1e-100"},
    {0x1.5ee2a2eb5a5c4p+53, "1.2345678901234568e+16"},
    {0x1.c6bf52633ffffp+49, "999999999999999.9"},
}};

}  // namespace

TEST_CASE("py_repr(double): matches CPython repr") {
  for (const Case& c : kCases) {
    CAPTURE(c.repr);
    CHECK(pantr::py_repr(c.value) == c.repr);
  }
}

TEST_CASE("py_repr(double): special values") {
  CHECK(pantr::py_repr(std::numeric_limits<double>::infinity()) == "inf");
  CHECK(pantr::py_repr(-std::numeric_limits<double>::infinity()) == "-inf");
  CHECK(pantr::py_repr(std::numeric_limits<double>::quiet_NaN()) == "nan");
  CHECK(pantr::py_repr(-std::numeric_limits<double>::quiet_NaN()) == "nan");
}

TEST_CASE("py_repr(shape): matches the spelling of a Python tuple of ints") {
  const std::vector<std::size_t> empty{};
  const std::vector<std::size_t> one{3};
  const std::vector<std::size_t> two{3, 4};
  const std::vector<std::size_t> three{0, 1, 12};
  CHECK(pantr::py_repr(std::span<const std::size_t>(empty)) == "()");
  CHECK(pantr::py_repr(std::span<const std::size_t>(one)) == "(3,)");
  CHECK(pantr::py_repr(std::span<const std::size_t>(two)) == "(3, 4)");
  CHECK(pantr::py_repr(std::span<const std::size_t>(three)) == "(0, 1, 12)");
}

TEST_CASE("py_repr_f32: spells a float as repr of the double of its shortest decimal string") {
  // Oracle: CPython's repr(float(str(numpy.float32(x)))) for the floats written in hexadecimal
  // (NumPy's str is the shortest string that reads back in float32), an implementation
  // independent of pantr's.
  struct F32Case {
    float value;
    const char* repr;
  };
  constexpr std::array<F32Case, 11> cases{{
      {0x1.99999ap-4F, "0.1"},
      {-0x1.99999ap-4F, "-0.1"},
      {0x1.4f8b58p-17F, "1e-05"},
      {0x1.000000p+24F, "16777216.0"},
      {0x1.fffffep+23F, "16777215.0"},
      {0x1.fffffep+127F, "3.4028235e+38"},
      {0x1.000000p-126F, "1.1754944e-38"},
      {0x1.0p+0F, "1.0"},
      {0x1.8p+1F, "3.0"},
      {0x1.6a09e6p+0F, "1.4142135"},
      {0x1.000000p+60F, "1.1529215e+18"},
  }};
  for (const F32Case& c : cases) {
    CAPTURE(c.repr);
    CHECK(pantr::py_repr_f32(c.value) == c.repr);
  }
  CHECK(pantr::py_repr_f32(std::numeric_limits<float>::infinity()) == "inf");
  CHECK(pantr::py_repr_f32(-std::numeric_limits<float>::infinity()) == "-inf");
  CHECK(pantr::py_repr_f32(std::numeric_limits<float>::quiet_NaN()) == "nan");
}

TEST_CASE("py_format_exp: matches Python's format(x, '.<p>e')") {
  // Oracle: CPython's format on the doubles written here.
  CHECK(pantr::py_format_exp(0.5617, 2) == "5.62e-01");
  CHECK(pantr::py_format_exp(1234567.0, 2) == "1.23e+06");
  CHECK(pantr::py_format_exp(1e-300, 2) == "1.00e-300");
  CHECK(pantr::py_format_exp(-0.0, 2) == "-0.00e+00");
  CHECK(pantr::py_format_exp(std::numeric_limits<double>::infinity(), 2) == "inf");
  CHECK(pantr::py_format_exp(-std::numeric_limits<double>::infinity(), 2) == "-inf");
  CHECK(pantr::py_format_exp(std::numeric_limits<double>::quiet_NaN(), 2) == "nan");
  CHECK_THROWS_AS(static_cast<void>(pantr::py_format_exp(1.0, 1000)), pantr::value_error);
}
