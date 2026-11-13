/// @file scalar.hpp
/// @brief The floating-point types pantr stores domain data in.
#pragma once

#include "pantr/config.hpp"

#include <concepts>

namespace pantr {

/// A storage type of pantr's domain values and tabulations: `float` or `double` (design §2.3).
/// Code that keys a decision on the format (a degree limit, an underflow bound, a dtype name)
/// relies on these two and no other.
template <class T>
concept StorageFloat = std::same_as<T, float> || std::same_as<T, double>;

}  // namespace pantr
