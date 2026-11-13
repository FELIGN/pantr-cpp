/// @file errors.hpp
/// @brief The exception hierarchy of the pantr C++ API.
///
/// Every exception pantr throws for a caller error derives from both a standard exception (so
/// generic handlers keep working) and `pantr::error` (so `catch (const pantr::error&)` catches
/// exactly pantr's own). The Python binding translates each class explicitly:
///
/// | C++ class                  | Standard base           | Python exception      |
/// |----------------------------|-------------------------|-----------------------|
/// | `value_error`              | `std::invalid_argument` | `ValueError`          |
/// | `type_error`               | `std::invalid_argument` | `TypeError`           |
/// | `index_error`              | `std::out_of_range`     | `IndexError`          |
/// | `state_error`              | `std::logic_error`      | `RuntimeError`        |
/// | `not_implemented`          | `std::logic_error`      | `NotImplementedError` |
/// | `capacity_error`           | `std::length_error`     | `ValueError`          |
///
/// `capacity_error` marks an implementation limit (an envelope a computation cannot exceed), as
/// opposed to an argument that is wrong in itself. Messages are part of the contract: the Python
/// suite matches them with regular expressions, so a message is changed only deliberately.
/// Design note: design/error_model.md.
#pragma once

#include "pantr/config.hpp"

#include <stdexcept>
#include <string>

namespace pantr {

/// Mixin base identifying an exception as pantr's own. Carries no data; the message lives in the
/// standard base.
class error {
 public:
  error(const error&) = default;
  error& operator=(const error&) = default;
  virtual ~error() = default;

 protected:
  error() = default;
};

/// An argument has the right type but a value pantr rejects.
class value_error : public std::invalid_argument, public error {
 public:
  using std::invalid_argument::invalid_argument;
};

/// An argument has a type (or dtype) pantr does not accept.
class type_error : public std::invalid_argument, public error {
 public:
  using std::invalid_argument::invalid_argument;
};

/// An index or position is outside the valid range.
class index_error : public std::out_of_range, public error {
 public:
  using std::out_of_range::out_of_range;
};

/// An object is in a state that forbids the operation (for example, a stale derived object).
class state_error : public std::logic_error, public error {
 public:
  using std::logic_error::logic_error;
};

/// The operation is valid in principle but not implemented for these arguments.
class not_implemented : public std::logic_error, public error {
 public:
  using std::logic_error::logic_error;
};

/// A request exceeds an implementation limit (a size, a degree or an exact-arithmetic envelope).
class capacity_error : public std::length_error, public error {
 public:
  using std::length_error::length_error;
};

}  // namespace pantr
