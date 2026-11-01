// CellTags and FacetTags: immutable registries. Oracles: hand-sorted arrays, the order Python's
// dict gives under reassignment, and repr() strings of CPython (recorded from `repr(str)`).
#include <doctest/doctest.h>

#include <pantr/grid/detail/py_text.hpp>
#include <pantr/grid/tags.hpp>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

using pantr::grid::CellTags;
using pantr::grid::FacetTags;
using pantr::grid::detail::py_repr_ints;
using pantr::grid::detail::py_repr_str;

namespace {

using I64 = std::int64_t;

std::vector<I64> vec(std::span<const I64> s) { return {s.begin(), s.end()}; }

}  // namespace

TEST_CASE("grid: py_repr_str: the spelling of repr(str)") {
  CHECK(py_repr_str("plain") == "'plain'");
  CHECK(py_repr_str("") == "''");
  CHECK(py_repr_str("a'b") == "\"a'b\"");
  CHECK(py_repr_str("a'b\"c") == "'a\\'b\"c'");
  CHECK(py_repr_str("'\"") == "'\\'\"'");
  CHECK(py_repr_str("\"") == "'\"'");
  CHECK(py_repr_str("'") == "\"'\"");
  CHECK(py_repr_str("a\nb") == "'a\\nb'");
  CHECK(py_repr_str("cr\r") == "'cr\\r'");
  CHECK(py_repr_str("tab\t") == "'tab\\t'");
  CHECK(py_repr_str("a\\b") == "'a\\\\b'");
  CHECK(py_repr_str(std::string("\x01", 1)) == "'\\x01'");
  CHECK(py_repr_str("\x7f") == "'\\x7f'");
  CHECK(py_repr_str(std::string("a\0b", 3)) == "'a\\x00b'");
  // Bytes from 0x80 on (here U+00E9, UTF-8 c3 a9) are copied as they are.
  CHECK(py_repr_str("\xc3\xa9") == "'\xc3\xa9'");
}

TEST_CASE("grid: py_repr_ints: the spelling of a tuple of ints") {
  const I64 none[1] = {0};
  CHECK(py_repr_ints(std::span<const I64>(none, 0)) == "()");
  CHECK(py_repr_ints(std::span<const I64>(none, 1)) == "(0,)");
  const I64 two[2] = {4, -1};
  CHECK(py_repr_ints(two) == "(4, -1)");
}

TEST_CASE("grid: CellTags: ids are stored sorted with their values moved along") {
  const CellTags empty(10);
  CHECK(empty.num_cells() == 10);
  CHECK(empty.size() == 0);
  const std::vector<I64> ids = {7, 2, 9, 0};
  const std::vector<I64> values = {70, 20, 90, 5};
  const CellTags tags = empty.set("inside", ids, values);
  const auto [got_ids, got_values] = tags.get("inside");
  CHECK(vec(got_ids) == std::vector<I64>{0, 2, 7, 9});
  CHECK(vec(got_values) == std::vector<I64>{5, 20, 70, 90});
  CHECK(tags.contains("inside"));
  CHECK_FALSE(tags.contains("outside"));
  CHECK(tags.size() == 1);
}

TEST_CASE("grid: CellTags: set and remove return new values and leave the receiver unchanged") {
  const CellTags a(5);
  const std::vector<I64> ids = {1, 3};
  const std::vector<I64> vals = {1, 1};
  const CellTags b = a.set("x", ids, vals);
  CHECK(a.size() == 0);
  CHECK(b.size() == 1);
  const CellTags c = b.remove("x");
  CHECK(b.size() == 1);
  CHECK(c.size() == 0);
  // The span handed out by `b` stays valid after `c` is gone: entries are shared, not moved.
  const auto [i1, v1] = b.get("x");
  const CellTags d = b.set("y", ids, vals);
  CHECK(d.get("x").first.data() == i1.data());
  CHECK(vec(v1) == std::vector<I64>{1, 1});
}

TEST_CASE("grid: CellTags: names keep insertion order, and reassigning a name keeps its place") {
  const std::vector<I64> ids = {0};
  const std::vector<I64> one = {1};
  const std::vector<I64> two = {2};
  CellTags t(4);
  t = t.set("a", ids, one);
  t = t.set("b", ids, one);
  t = t.set("c", ids, one);
  CHECK(t.names() == std::vector<std::string>{"a", "b", "c"});
  t = t.set("a", ids, two);
  CHECK(t.names() == std::vector<std::string>{"a", "b", "c"});
  CHECK(t.get("a").second[0] == 2);
  t = t.remove("b");
  CHECK(t.names() == std::vector<std::string>{"a", "c"});
  t = t.set("b", ids, one);
  CHECK(t.names() == std::vector<std::string>{"a", "c", "b"});
}

TEST_CASE("grid: CellTags: an empty tag is allowed") {
  const std::vector<I64> none;
  const CellTags t = CellTags(3).set("nothing", none, none);
  CHECK(t.get("nothing").first.empty());
  CHECK(CellTags(0).set("z", none, none).size() == 1);
}

TEST_CASE("grid: CellTags: validation messages") {
  CHECK_THROWS_WITH_AS(CellTags(-1), "num_cells must be >= 0; got -1.", pantr::value_error);
  const CellTags t(5);
  const std::vector<I64> ids_out = {0, 5};
  const std::vector<I64> two = {1, 1};
  CHECK_THROWS_WITH_AS(static_cast<void>(t.set("x", ids_out, two)),
                       "cell ids must be in [0, 5); got range [0, 5].", pantr::value_error);
  const std::vector<I64> ids_neg = {-3, 2};
  CHECK_THROWS_WITH_AS(static_cast<void>(t.set("x", ids_neg, two)),
                       "cell ids must be in [0, 5); got range [-3, 2].", pantr::value_error);
  const std::vector<I64> dup = {2, 4, 2};
  const std::vector<I64> three = {1, 1, 1};
  CHECK_THROWS_WITH_AS(static_cast<void>(t.set("it's", dup, three)),
                       "cell ids for tag \"it's\" must be unique; got duplicates.",
                       pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(t.set("x", dup, two)),
                       "cell ids for tag 'x' must be unique; got duplicates.", pantr::value_error);
  const std::vector<I64> ids = {1, 2};
  const std::vector<I64> wrong = {1, 1, 1};
  CHECK_THROWS_WITH_AS(static_cast<void>(t.set("x", ids, wrong)),
                       "values must be a scalar or have length 2; got length 3.",
                       pantr::value_error);
  // The range check precedes the uniqueness check, which precedes the length check.
  CHECK_THROWS_WITH_AS(static_cast<void>(t.set("x", std::vector<I64>{9, 9}, wrong)),
                       "cell ids must be in [0, 5); got range [9, 9].", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(t.set("x", dup, std::vector<I64>{})),
                       "cell ids for tag 'x' must be unique; got duplicates.", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(t.get("nope")), "no tag named 'nope'.",
                       pantr::index_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(t.remove("nope")), "no tag named 'nope'.",
                       pantr::index_error);
}

TEST_CASE("grid: FacetTags: keys are stored sorted lexicographically by (cell, facet)") {
  const FacetTags empty(6, 4);
  CHECK(empty.num_cells() == 6);
  CHECK(empty.facets_per_cell() == 4);
  // Rows (cid, lfid): (3,1) (0,2) (3,0) (0,0) (5,3).
  const std::vector<I64> keys = {3, 1, 0, 2, 3, 0, 0, 0, 5, 3};
  const std::vector<I64> values = {31, 2, 30, 0, 53};
  const FacetTags t = empty.set("wall", keys, values);
  const auto [k, v] = t.get("wall");
  CHECK(vec(k) == std::vector<I64>{0, 0, 0, 2, 3, 0, 3, 1, 5, 3});
  CHECK(vec(v) == std::vector<I64>{0, 2, 30, 31, 53});
  CHECK(empty.size() == 0);
}

TEST_CASE("grid: FacetTags: names, replacement in place and removal") {
  const std::vector<I64> keys = {0, 0};
  const std::vector<I64> one = {1};
  const std::vector<I64> two = {2};
  FacetTags t(2, 2);
  t = t.set("a", keys, one);
  t = t.set("b", keys, one);
  t = t.set("a", keys, two);
  CHECK(t.names() == std::vector<std::string>{"a", "b"});
  CHECK(t.get("a").second[0] == 2);
  t = t.remove("a");
  CHECK(t.names() == std::vector<std::string>{"b"});
  CHECK_THROWS_AS(static_cast<void>(t.remove("a")), pantr::index_error);
}

TEST_CASE("grid: FacetTags: validation messages") {
  CHECK_THROWS_WITH_AS(FacetTags(-1, 2), "num_cells must be >= 0; got -1.", pantr::value_error);
  CHECK_THROWS_WITH_AS(FacetTags(3, 0), "facets_per_cell must be >= 1; got 0.", pantr::value_error);
  CHECK_THROWS_AS(FacetTags(-1, 0), pantr::value_error);
  const FacetTags t(3, 4);
  const std::vector<I64> v1 = {1};
  CHECK_THROWS_WITH_AS(static_cast<void>(t.set("x", std::vector<I64>{3, 0}, v1)),
                       "facet cell ids must be in [0, 3); got range [3, 3].", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(t.set("x", std::vector<I64>{0, -1, 2, 0}, v1)),
                       "local facet ids must be in [0, 4); got range [-1, 0].", pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(t.set("x", std::vector<I64>{0, 4}, v1)),
                       "local facet ids must be in [0, 4); got range [4, 4].", pantr::value_error);
  const std::vector<I64> dup = {1, 2, 0, 0, 1, 2};
  CHECK_THROWS_WITH_AS(static_cast<void>(t.set("w", dup, std::vector<I64>{1, 1, 1})),
                       "facet keys for tag 'w' must be unique; got duplicates.",
                       pantr::value_error);
  // The same cell with different facets, and the same facet of different cells, are distinct.
  const std::vector<I64> distinct = {1, 2, 1, 3, 2, 2};
  const std::vector<I64> v3 = {1, 1, 1};
  const std::vector<I64> v2 = {1, 1};
  CHECK_NOTHROW(static_cast<void>(t.set("w", distinct, v3)));
  CHECK_THROWS_WITH_AS(static_cast<void>(t.set("w", std::vector<I64>{1, 2}, v2)),
                       "values must be a scalar or have length 1; got length 2.",
                       pantr::value_error);
  CHECK_THROWS_AS(static_cast<void>(t.set("w", std::vector<I64>{1, 2, 3}, v1)), pantr::value_error);
  CHECK_THROWS_WITH_AS(static_cast<void>(t.get("zz")), "no tag named 'zz'.", pantr::index_error);
}
