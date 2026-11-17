/// @file py_text.hpp
/// @brief Python spellings of a string and of a tuple of ints, for messages that interpolate them.
#pragma once

#include "pantr/config.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace pantr::grid::detail {

/// Spell a tuple of ints as Python spells `repr(tuple)`: `()`, `(3,)`, `(3, 4)`.
///
/// Args:
///     xs (std::span<const std::int64_t>): The entries.
///
/// Returns:
///     std::string: The spelling.
[[nodiscard]] inline std::string py_repr_ints(std::span<const std::int64_t> xs) {
  std::string out = "(";
  for (std::size_t i = 0; i < xs.size(); ++i) {
    if (i > 0) {
      out += ", ";
    }
    out += std::to_string(xs[i]);
  }
  if (xs.size() == 1) {
    out += ',';
  }
  out += ')';
  return out;
}

/// Spell a string as CPython spells `repr(str)`.
///
/// Single quotes, unless the string contains `'` and no `"`, in which case double quotes. The
/// backslash, the chosen quote, `\n`, `\r` and `\t` are escaped, and every other byte below 0x20
/// and 0x7f as `\xNN` (lower-case hex). Bytes from 0x80 on are copied as they are: CPython
/// escapes the non-printable code points among them (for example U+0080 as `\x80`), which this
/// does not, since telling them apart needs the Unicode tables. A name in a message is ASCII in
/// practice.
///
/// Args:
///     s (std::string_view): The string (UTF-8).
///
/// Returns:
///     std::string: The quoted, escaped spelling.
[[nodiscard]] inline std::string py_repr_str(std::string_view s) {
  const bool has_single = s.find('\'') != std::string_view::npos;
  const bool has_double = s.find('"') != std::string_view::npos;
  const char quote = (has_single && !has_double) ? '"' : '\'';
  static constexpr char hex[] = "0123456789abcdef";
  std::string out;
  out += quote;
  for (const char ch : s) {
    const auto byte = static_cast<unsigned char>(ch);
    if (ch == quote || ch == '\\') {
      out += '\\';
      out += ch;
    } else if (ch == '\n') {
      out += "\\n";
    } else if (ch == '\r') {
      out += "\\r";
    } else if (ch == '\t') {
      out += "\\t";
    } else if (byte < 0x20 || byte == 0x7f) {
      out += "\\x";
      out += hex[byte >> 4];
      out += hex[byte & 0x0f];
    } else {
      out += ch;
    }
  }
  out += quote;
  return out;
}

}  // namespace pantr::grid::detail
