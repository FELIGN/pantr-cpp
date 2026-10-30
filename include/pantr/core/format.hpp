/// @file format.hpp
/// @brief Python-compatible text for values interpolated into error messages.
///
/// The Python suite matches pantr's error messages with regular expressions, and several messages
/// interpolate a float or an array shape. These functions spell them exactly as Python's `repr`
/// does, so a message built in C++ reads the same as the one the Python implementation built.
#pragma once

#include "pantr/config.hpp"
#include "pantr/core/errors.hpp"

#include <charconv>
#include <cmath>
#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <system_error>

namespace pantr {

/// Spell a double as Python's `repr(float)` does.
///
/// The digits are the shortest decimal string that reads back to the same double (as
/// `std::to_chars` produces them); the layout follows CPython's `float_repr`: with the value
/// written as `0.D * 10**decpt`, positional notation when `-4 < decpt <= 16`, otherwise
/// exponential with a signed, at-least-two-digit exponent; an integral positional value keeps a
/// trailing `.0`. Special values are `nan` (whatever the sign bit), `inf`, `-inf`; zero keeps its
/// sign (`-0.0`).
///
/// Args:
///     x (double): The value.
///
/// Returns:
///     std::string: The spelling, e.g. `3.0`, `0.1`, `1e-05`, `1.5e+16`, `-0.0`, `inf`, `nan`.
///
/// Example:
///     py_repr(1e-5) == "1e-05"; py_repr(100.0) == "100.0"; py_repr(1e16) == "1e+16"
[[nodiscard]] inline std::string py_repr(double x) {
  if (std::isnan(x)) {
    return "nan";
  }
  if (std::isinf(x)) {
    return x > 0 ? "inf" : "-inf";
  }
  // Shortest round-trip digits in scientific form: [-]d[.ddd]e(+|-)dd[d]. The longest such
  // double is 24 characters, so the conversion into 64 bytes cannot run out of room.
  char buf[64];
  const auto res = std::to_chars(buf, buf + sizeof(buf), x, std::chars_format::scientific);
  const std::string_view sci(buf, static_cast<std::size_t>(res.ptr - buf));

  std::string_view rest = sci;
  const bool negative = !rest.empty() && rest.front() == '-';
  if (negative) {
    rest.remove_prefix(1);
  }
  const std::size_t e_pos = rest.find('e');
  std::string digits;
  for (const char c : rest.substr(0, e_pos)) {
    if (c != '.') {
      digits.push_back(c);
    }
  }
  const std::string_view exp_text = rest.substr(e_pos + 1);
  int exponent = 0;
  // from_chars does not accept a leading '+'.
  const std::string_view exp_digits =
      (exp_text.front() == '+' || exp_text.front() == '-') ? exp_text.substr(1) : exp_text;
  std::from_chars(exp_digits.data(), exp_digits.data() + exp_digits.size(), exponent);
  if (exp_text.front() == '-') {
    exponent = -exponent;
  }

  // The value is 0.<digits> * 10**decpt.
  const int decpt = exponent + 1;
  const int ndigits = static_cast<int>(digits.size());
  std::string out = negative ? "-" : "";
  if (decpt <= -4 || decpt > 16) {
    out += digits.front();
    if (ndigits > 1) {
      out += '.';
      out.append(digits, 1);
    }
    const int e = decpt - 1;
    out += e < 0 ? "e-" : "e+";
    const int abs_e = e < 0 ? -e : e;
    if (abs_e < 10) {
      out += '0';
    }
    out += std::to_string(abs_e);
  } else if (decpt <= 0) {
    out += "0.";
    out.append(static_cast<std::size_t>(-decpt), '0');
    out += digits;
  } else if (decpt >= ndigits) {
    out += digits;
    out.append(static_cast<std::size_t>(decpt - ndigits), '0');
    out += ".0";
  } else {
    out.append(digits, 0, static_cast<std::size_t>(decpt));
    out += '.';
    out.append(digits, static_cast<std::size_t>(decpt));
  }
  return out;
}

/// Spell a double as Python's `format(x, '.<precision>g')` or `format(x, '.<precision>f')` does.
///
/// `std::to_chars` in `general` or `fixed` form with a precision follows C's `%g` and `%f`, which
/// Python's `g` and `f` presentation types also follow (exponent threshold, trailing zeros removed
/// by `g`, a signed at-least-two-digit exponent), and is locale-independent. Special values are
/// spelled as Python spells them: `inf`, `-inf`, and `nan` whatever the sign bit.
///
/// Args:
///     x (double): The value.
///     precision (int): Significant digits (`g`) or digits after the point (`f`), `>= 0`.
///     fixed (bool): `true` for the `f` presentation type, `false` for `g`.
///
/// Returns:
///     std::string: The spelling, e.g. `1.23e-05` for `(1.2345e-5, 3, false)`, `8` for
///     `(7.5, 0, true)`.
[[nodiscard]] inline std::string py_format(double x, int precision, bool fixed) {
  if (std::isnan(x)) {
    return "nan";
  }
  if (std::isinf(x)) {
    return x > 0 ? "inf" : "-inf";
  }
  // A fixed spelling of the largest double has 309 digits before the point.
  char buf[400];
  const auto fmt = fixed ? std::chars_format::fixed : std::chars_format::general;
  const auto res = std::to_chars(buf, buf + sizeof(buf), x, fmt, precision);
  if (res.ec != std::errc{}) {
    // Only a fixed spelling with a precision of about 90 or more digits runs out of room.
    throw value_error("py_format: precision " + std::to_string(precision) + " is too large");
  }
  return {buf, static_cast<std::size_t>(res.ptr - buf)};
}

/// Spell an array shape as Python spells the tuple `ndarray.shape`.
///
/// Args:
///     shape (std::span<const std::size_t>): The extents.
///
/// Returns:
///     std::string: `()`, `(3,)` or `(3, 4)`.
[[nodiscard]] inline std::string py_repr(std::span<const std::size_t> shape) {
  std::string out = "(";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) {
      out += ", ";
    }
    out += std::to_string(shape[i]);
  }
  if (shape.size() == 1) {
    out += ',';
  }
  out += ')';
  return out;
}

}  // namespace pantr
