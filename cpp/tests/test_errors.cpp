// The exception hierarchy: each class is catchable as its standard base and as pantr::error.
#include <doctest/doctest.h>

#include <pantr/core/errors.hpp>

#include <stdexcept>
#include <string>
#include <type_traits>

namespace {

template <class E, class Std>
void check_class(const char* msg) {
  static_assert(std::is_base_of_v<Std, E>);
  static_assert(std::is_base_of_v<pantr::error, E>);
  try {
    throw E(msg);
  } catch (const Std& e) {
    CHECK(std::string(e.what()) == msg);
  }
  try {
    throw E(msg);
  } catch (const pantr::error& e) {
    const auto* std_view = dynamic_cast<const std::exception*>(&e);
    REQUIRE(std_view != nullptr);
    CHECK(std::string(std_view->what()) == msg);
  }
}

}  // namespace

TEST_CASE("errors: every class derives from its standard base and pantr::error") {
  check_class<pantr::value_error, std::invalid_argument>("bad value");
  check_class<pantr::type_error, std::invalid_argument>("bad type");
  check_class<pantr::index_error, std::out_of_range>("bad index");
  check_class<pantr::state_error, std::logic_error>("stale");
  check_class<pantr::not_implemented, std::logic_error>("later");
  check_class<pantr::capacity_error, std::length_error>("too big");
}

TEST_CASE("errors: a standard exception that is not pantr's is not a pantr::error") {
  bool caught_as_pantr = false;
  try {
    throw std::invalid_argument("foreign");
  } catch (const pantr::error&) {
    caught_as_pantr = true;
  } catch (const std::invalid_argument&) {
  }
  CHECK_FALSE(caught_as_pantr);
}

TEST_CASE("errors: pantr::error cannot be thrown on its own") {
  static_assert(!std::is_constructible_v<pantr::error>);
  CHECK(true);
}
